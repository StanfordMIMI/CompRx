import os

import hydra
import pyrootutils
import torch
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich import print
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric

from comprx.utils.cls import training_epoch, validation_epoch
from comprx.utils.extras import sanitize_dataloader_kwargs, set_seed
from comprx.utils.lr_schedulers import CosineScheduler

# Set the project root
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
config_dir = os.path.join(root, "configs")

# Register configuration resolvers
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base="1.2", config_path=config_dir, config_name="train.yaml")
def main(cfg: DictConfig):
    print(f"=> Starting [experiment={cfg['task_name']}]")
    print("=> Initializing Hydra configuration")
    cfg = instantiate(cfg)

    seed = cfg.get("seed", None)
    if seed is not None:
        set_seed(seed)

    # Setup accelerator
    is_logging = cfg.get("logger", None) is not None
    print(f"=> Instantiate accelerator [logging={is_logging}]")
    logger_name = "wandb" if is_logging else None
    logger_kwargs = {"wandb": cfg.get("logger", None)}

    assert cfg.get("mixed_precision", None) in ["bf16", "fp16", "no"]
    gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 1)
    accelerator = Accelerator(
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=gradient_accumulation_steps,
            adjust_scheduler=False,
        ),
        mixed_precision=cfg.get("mixed_precision", None),
        log_with=logger_name,
        split_batches=True,
    )

    print(f"=> Mixed precision: {accelerator.mixed_precision}")
    accelerator.init_trackers("comprx", config=cfg, init_kwargs=logger_kwargs)
    device = accelerator.device

    # instantiate dataloaders
    print(f"=> Instantiating train dataloader [device={device}]")
    train_dataloader = DataLoader(**sanitize_dataloader_kwargs(cfg["dataloader"]["train"]))

    print(f"=> Instantiating valid dataloader [device={device}]")
    valid_dataloader = DataLoader(**sanitize_dataloader_kwargs(cfg["dataloader"]["valid"]))

    # create the model
    print(f"=> Creating model [device={device}]")
    model = cfg["model"]
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # build optimizer from a partial optimizer
    print(f"=> Instantiating the optimizer [device={device}]")
    params = list(filter(lambda p: p.requires_grad, model.parameters()))

    lr, batch_size = cfg["optimizer"].keywords["lr"], cfg["dataloader"]["train"]["batch_size"]
    init_lr = lr * batch_size * gradient_accumulation_steps / 256
    optimizer = cfg["optimizer"](params, lr=init_lr)

    # learning rate scheduler
    print(f"=> Instantiating LR scheduler [device={device}]")
    scheduler = CosineScheduler(optimizer, cfg["max_epoch"])

    # loss function
    criterion = cfg["criterion"]

    # prepare the components for multi-gpu/mixed precision training
    (
        train_dataloader,
        valid_dataloader,
        model,
        optimizer,
        scheduler,
        criterion,
    ) = accelerator.prepare(
        train_dataloader,
        valid_dataloader,
        model,
        optimizer,
        scheduler,
        criterion,
    )

    accelerator.register_for_checkpointing(scheduler)

    # prepare the metrics
    default_metrics = accelerator.prepare(*[MeanMetric() for _ in range(3)])
    if len(cfg["metrics"]) > 0:
        names, metrics = list(zip(*cfg["metrics"]))
        metrics = list(zip(names, accelerator.prepare(*metrics)))
    else:
        metrics = []

    if len(cfg["metrics_slice"]) > 0:
        names, group_ids, metrics_slice = list(zip(*cfg["metrics_slice"]))
        slice_metrics = list(
            zip(names, [g["group_id"] for g in group_ids], accelerator.prepare(*metrics_slice))
        )
    else:
        slice_metrics = []

    # resume from checkpoint
    start_epoch = cfg["start_epoch"]
    if cfg["resume_from_ckpt"] is not None:
        accelerator.load_state(cfg["resume_from_ckpt"])
        custom_ckpt = torch.load(os.path.join(cfg["resume_from_ckpt"], "custom_checkpoint_0.pkl"))
        start_epoch = custom_ckpt["last_epoch"]

    # setup metrics
    max_metric = None

    print(f"=> Starting model training [epochs={cfg['max_epoch']}]")
    for epoch in range(start_epoch, cfg["max_epoch"]):
        # train one epoch
        training_epoch(
            cfg=cfg,
            epoch=epoch,
            accelerator=accelerator,
            dataloader=train_dataloader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            default_metrics=default_metrics,
            classif_metrics=metrics,
        )

        accelerator.wait_for_everyone()
        # adjust the learning rate per epoch
        scheduler.step()

        # evaluate the model
        metric = validation_epoch(
            cfg=cfg,
            epoch=epoch,
            accelerator=accelerator,
            dataloader=valid_dataloader,
            model=model,
            criterion=criterion,
            default_metrics=default_metrics,
            classif_metrics=metrics,
            slice_metrics=slice_metrics,
        )

        # save the best model
        if max_metric is None or metric > max_metric:
            max_metric = metric
            try:
                accelerator.save_state(os.path.join(cfg["ckpt_dir"], "best.pt"))
            except Exception as e:
                print(e)

        # save checkpoint
        if (epoch + 1) % cfg["ckpt_every_n_epochs"] == 0 and not cfg.get("disable_ckpts", False):
            print(f"=> Saving checkpoint [epoch={epoch}]")
            try:
                accelerator.save_state(os.path.join(cfg["ckpt_dir"], f"epoch-{epoch:04d}.pt"))
            except Exception as e:
                print(e)

    # save last model
    if not cfg.get("disable_ckpts", False):
        print(f"=> Saving last checkpoint [epoch={epoch}]")
        accelerator.save_state(os.path.join(cfg["ckpt_dir"], "last.pt"))

    print(f"=> Finished model training [epochs={cfg['max_epoch']}, metric={max_metric}]")
    accelerator.end_training()


if __name__ == "__main__":
    main()
