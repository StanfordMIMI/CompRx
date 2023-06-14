import os

import hydra
import pyrootutils
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import GradientAccumulationPlugin
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich import print
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric

from comprx.utils.extras import sanitize_dataloader_kwargs, set_seed
from comprx.utils.vae.litema import LitEma, ema_scope
from comprx.utils.vae.train_components import training_epoch, validation_epoch

# Set the project root
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
config_dir = os.path.join(root, "configs")

# Register configuration resolvers
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base="1.2", config_path=config_dir, config_name="train_vae.yaml")
def main(cfg: DictConfig):
    # Instantiating config
    print(f"=> Starting [experiment={cfg.task_name}]")
    cfg = instantiate(cfg)

    # Seeding
    if cfg.get("seed", None) is not None:
        print(f"=> Setting seed [seed={cfg.seed}]")
        set_seed(cfg.seed)

    torch.backends.cuda.matmul.allow_tf32 = True

    # Setup accelerator
    logger_kwargs = cfg.get("logger", None)
    is_logging = bool(logger_kwargs)
    print(f"=> Instantiate accelerator [logging={is_logging}]")

    gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 1)
    accelerator = Accelerator(
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=gradient_accumulation_steps,
            adjust_scheduler=False,
        ),
        mixed_precision=cfg.get("mixed_precision", None),
        log_with="wandb" if is_logging else None,
        split_batches=True,
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=True,
            )
        ],
    )
    accelerator.init_trackers("comprx", config=cfg, init_kwargs={"wandb": logger_kwargs})

    # Determine the mode
    print(f"=> Mixed precision: {accelerator.mixed_precision}")

    inference_mode = cfg.get("inference", False)
    print(f"=> Running in inference mode: {inference_mode}")

    print(f"=> Instantiating train dataloader [device={accelerator.device}]")
    train_dataloader = DataLoader(**sanitize_dataloader_kwargs(cfg["dataloader"]["train"]))

    print(f"=> Instantiating valid dataloader [device={accelerator.device}]")
    valid_dataloader = DataLoader(**sanitize_dataloader_kwargs(cfg["dataloader"]["valid"]))

    # Create loss function
    criterion = cfg.criterion
    discriminator_iter_start = criterion.discriminator_iter_start

    # Create model
    model = cfg.model
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Set up exponential moving average (EMA) parameter tracking
    use_ema = cfg.get("ema_decay", None) is not None
    if use_ema:
        assert 0.0 < cfg.ema_decay < 1.0
        model_ema = LitEma([model, criterion], decay=cfg.ema_decay)
        print(f"Tracking {len(list(model_ema.buffers()))} buffers for EMA computations.")

    # Create two optimizers: one for the autoencoder and one for the discriminator
    print(f"=> Instantiating the optimizer [device={accelerator.device}]")

    batch_size, lr = cfg.batch_size, cfg.base_learning_rate
    lr = gradient_accumulation_steps * batch_size * lr

    ae_params = (
        list(model.encoder.parameters())
        + list(model.decoder.parameters())
        + list(model.quant_conv.parameters())
        + list(model.post_quant_conv.parameters())
    )
    if criterion.learn_logvar:
        ae_params.append(criterion.logvar)
    opt_ae = torch.optim.Adam(ae_params, lr=lr, betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(criterion.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

    # Prepare components for multi-gpu/mixed precision training
    (train_dataloader, valid_dataloader, model, opt_ae, opt_disc, criterion) = accelerator.prepare(
        train_dataloader,
        valid_dataloader,
        model,
        opt_ae,
        opt_disc,
        criterion,
    )

    # Create metrics: aeloss, discloss, recloss, data(time), batch(time)
    default_metrics = accelerator.prepare(*[MeanMetric() for _ in range(5)])
    if len(cfg["metrics"]) > 0:
        names, metrics = list(zip(*cfg["metrics"]))
        metrics = list(zip(names, accelerator.prepare(*metrics)))
    else:
        metrics = []

    # Resume from checkpoint
    start_epoch = cfg.start_epoch
    if cfg.resume_from_ckpt is not None:
        accelerator.load_state(cfg.resume_from_ckpt)

    options = {
        "max_epoch": cfg["max_epoch"],
        "is_logging": is_logging,
        "log_every_n_steps": cfg["log_every_n_steps"],
        "ckpt_every_n_steps": cfg["ckpt_every_n_steps"],
        "ckpt_dir": cfg["ckpt_dir"],
        "fast_dev_run": cfg["fast_dev_run"],
    }

    if inference_mode:
        print("=> Inference mode: saving latents")
        model.requires_grad_(False)
        accelerator.wait_for_everyone()
        with torch.no_grad():
            for step, batch in enumerate(train_dataloader):
                if accelerator.is_main_process:
                    print(f"=> Step: {step}/{len(train_dataloader)}")

                z = model.encode(batch["img"]).mode()
                z = z.cpu().to(dtype=torch.float32)

                for arr, path in zip(z, batch["txt"]):
                    out_path = (
                        f"{cfg.paths.mg_data_dir}/mg-{batch['group_id'].item()}/latent/learned"
                    )
                    os.makedirs(out_path, exist_ok=True)
                    output_path = os.path.join(out_path, path + ".pt")
                    torch.save(arr, output_path)

            for step, batch in enumerate(valid_dataloader):
                if accelerator.is_main_process:
                    print(f"=> Step: {step}/{len(valid_dataloader)}")

                z = model.encode(batch["img"]).mode()
                z = z.cpu().to(dtype=torch.float32)

                for arr, path in zip(z, batch["txt"]):
                    out_path = (
                        f"{cfg.paths.mg_data_dir}/mg-{batch['group_id'].item()}/latent/learned"
                    )
                    os.makedirs(out_path, exist_ok=True)
                    output_path = os.path.join(out_path, path + ".pt")
                    torch.save(arr, output_path)
    else:
        print(f"=> Starting model training [epochs={cfg['max_epoch']}]")
        min_loss = None
        global_step = cfg.get("global_step", 0)
        for epoch in range(start_epoch, cfg["max_epoch"]):
            global_step = training_epoch(
                options=options,
                epoch=epoch,
                global_step=global_step,
                accelerator=accelerator,
                dataloader=train_dataloader,
                model=model,
                criterion=criterion,
                discriminator_iter_start=discriminator_iter_start,
                default_metrics=default_metrics,
                rec_metrics=metrics,
                optimizer_ae=opt_ae,
                optimizer_disc=opt_disc,
            )

            accelerator.wait_for_everyone()
            if use_ema:
                model_ema([model, criterion])

            loss = validation_epoch(
                options=options,
                epoch=epoch,
                accelerator=accelerator,
                dataloader=valid_dataloader,
                model=model,
                criterion=criterion,
                default_metrics=default_metrics,
                rec_metrics=metrics,
                global_step=global_step,
            )

            if use_ema:
                with ema_scope(use_ema, model_ema, [model, criterion]):
                    validation_epoch(
                        options=options,
                        epoch=epoch,
                        accelerator=accelerator,
                        dataloader=valid_dataloader,
                        model=model,
                        criterion=criterion,
                        default_metrics=default_metrics,
                        rec_metrics=metrics,
                        global_step=global_step,
                        postfix="_ema",
                    )

            # save the best model
            if min_loss is None or loss < min_loss:
                try:
                    accelerator.save_state(os.path.join(cfg.ckpt_dir, "best.pt"))
                except Exception as e:
                    print(e)
                min_loss = loss

            # save checkpoint
            if (epoch + 1) % cfg.get("ckpt_every_n_epochs", 1) == 0:
                print(f"=> Saving checkpoint [epoch={epoch}]")
                try:
                    accelerator.save_state(os.path.join(cfg.ckpt_dir, f"epoch-{epoch:04d}.pt"))
                except Exception as e:
                    print(e)

        # save last model
        accelerator.save_state(os.path.join(cfg.ckpt_dir, "last.pt"))

        print(f"=> Finished model training [epochs={cfg['max_epoch']}, metric={min_loss}]")
        accelerator.end_training()


if __name__ == "__main__":
    main()
