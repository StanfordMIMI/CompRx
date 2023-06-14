import os

import hydra
import pyrootutils
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich import print
from torch.utils.data import DataLoader

from comprx.utils.extras import sanitize_dataloader_kwargs, set_seed

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

    print("=> Instantiating accelerator")
    accelerator = Accelerator(
        mixed_precision=cfg.get("mixed_precision", None),
        log_with=None,
        split_batches=True,
    )
    print(f"=> Mixed precision: {accelerator.mixed_precision}")

    print(f"=> Instantiating train dataloader [device={accelerator.device}]")
    train_dataloader = DataLoader(**sanitize_dataloader_kwargs(cfg["dataloader"]["train"]))

    print(f"=> Instantiating valid dataloader [device={accelerator.device}]")
    valid_dataloader = DataLoader(**sanitize_dataloader_kwargs(cfg["dataloader"]["valid"]))

    # Create model
    model = cfg.model

    # Prepare components
    train_dataloader, valid_dataloader, model = accelerator.prepare(
        train_dataloader, valid_dataloader, model
    )

    # Resume from checkpoint
    assert isinstance(cfg.resume_from_ckpt, str), "Please set the resume_from_ckpt in the config"
    accelerator.load_state(cfg.resume_from_ckpt)

    print("=> Inference mode: saving latents")
    model.requires_grad_(False)
    accelerator.wait_for_everyone()

    output_dir = cfg.paths.get("inference_output_dir", None)
    assert os.path.exists(output_dir), f"Path {output_dir} does not exist"

    # Run inference for both the training dataset and the validation dataset
    with torch.no_grad():
        print("=> Starting train dataloader...")
        for step, batch in enumerate(train_dataloader):
            if accelerator.is_main_process:
                print(f"=> Step: {step}/{len(train_dataloader)}")

            z = model.encode(batch["img"]).mode()
            z = z.cpu().to(dtype=torch.float32)

            for arr, fname in zip(z, batch["txt"]):
                output_path = os.path.join(output_dir, fname + ".pt")
                torch.save(arr, output_path)

        print("=> Starting valid dataloader...")
        for step, batch in enumerate(valid_dataloader):
            if accelerator.is_main_process:
                print(f"=> Step: {step}/{len(valid_dataloader)}")

            z = model.encode(batch["img"]).mode()
            z = z.cpu().to(dtype=torch.float32)

            for arr, fname in zip(z, batch["txt"]):
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, fname + ".pt")
                torch.save(arr, output_path)


if __name__ == "__main__":
    main()
