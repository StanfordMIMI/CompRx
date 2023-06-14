import ast
import json
import os
import random

import hydra
import numpy as np
import pandas as pd
import pyrootutils
import torch
import torchvision
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich import print
from torch.utils.data import DataLoader
from torchvision.transforms import Resize

from comprx.models import AutoencoderKL, AutoencoderVQ
from comprx.utils.extras import sanitize_dataloader_kwargs, set_seed

NUM_SAMPLES = 1000

# Set the project root
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
config_dir = os.path.join(root, "configs")

# Register configuration resolvers
OmegaConf.register_new_resolver("eval", eval)


def get_dataset_cat(dataset_ids, fine_grained):
    if dataset_ids == [1, 2, 3, 4, 5, 6, 7, 8]:
        dataset_cat = "all"
    elif dataset_ids == [1, 2, 3, 4, 5, 6]:
        dataset_cat = "mg"
    elif dataset_ids == [7, 8]:
        dataset_cat = "cxr"
    elif dataset_ids == [9, 10]:
        dataset_cat = "msk"
    elif dataset_ids == [10] and fine_grained is False:
        dataset_cat = "wrist"
    elif dataset_ids == [10] and fine_grained:
        dataset_cat = "fg_msk"
    else:
        raise ValueError

    return dataset_cat


def get_bbox(dataset_cat, img_size, txt):
    if dataset_cat == "fg_msk":
        df = pd.read_csv("/fsx/aimi/comprx/msk-2/splits/bbox.csv")
        id2box = dict(zip(df["id"], df["bounding_box"]))
        id2shape = dict(zip(df["id"], df["shape"]))
    else:
        raise ValueError

    box = ast.literal_eval(id2box[txt[0]])
    div = np.array(ast.literal_eval(id2shape[txt[0]])) / img_size
    box = (
        round(box[0] / div[0]),
        round(box[1] / div[1]),
        round(box[2] / div[0]),
        round(box[3] / div[1]),
    )

    return box


@hydra.main(version_base="1.2", config_path=config_dir, config_name="train_vae.yaml")
def main(cfg: DictConfig):
    # Instantiating config
    print(f"=> Starting [experiment={cfg.task_name}]")
    cfg = instantiate(cfg)

    # Seeding
    print(f"=> Setting seed [seed={cfg.seed}]")
    set_seed(cfg.seed)
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    torch.backends.cuda.matmul.allow_tf32 = True

    model_type = cfg.resume_from_ckpt.split("-")[0]
    if model_type == "kl":  # e.g. kl-f8
        print(f"=> Generating metrics for Stable Diffusion model {cfg.resume_from_ckpt}")
        mode = "sd_kl"
    elif model_type == "vq":  # e.g. vq-f8
        print(f"=> Generating metrics for Stable Diffusion model {cfg.resume_from_ckpt}")
        mode = "sd_vq"
    elif model_type in ["bicubic", "bilinear", "nearest"]:  # e.g. bicubic-4x
        print(f"=> Generating metrics for Interpolation {cfg.resume_from_ckpt}")
        mode = model_type
    elif model_type == "ours":  # e.g. ours-4x1
        print(f"=> Generating metrics for our VAEs {cfg.resume_from_ckpt}")
        mode = "ours"
    else:
        raise ValueError

    print("=> Instantiating valid dataloader")
    valid_dataloader = DataLoader(
        **sanitize_dataloader_kwargs(cfg["dataloader"]["valid"]),
        worker_init_fn=seed_worker,
        generator=g,
    )

    dataset_ids = cfg.dataset_ids
    dataset_cat = get_dataset_cat(dataset_ids, cfg.fine_grained)

    if mode == "sd_kl":
        conf = OmegaConf.load(f"/admin/home-mayavarma/vae-weights/{cfg.resume_from_ckpt}.yaml")
        model = AutoencoderKL(
            ddconfig=conf.model.params.ddconfig,
            embed_dim=conf.model.params.embed_dim,
            ckpt_path=f"/admin/home-mayavarma/vae-weights/{cfg.resume_from_ckpt}.ckpt",
        ).cuda()
        model.requires_grad_(False)
    elif mode == "sd_vq":
        conf = OmegaConf.load(f"/admin/home-mayavarma/vae-weights/{cfg.resume_from_ckpt}.yaml")
        model = AutoencoderVQ(
            ddconfig=conf.model.params.ddconfig,
            embed_dim=conf.model.params.embed_dim,
            n_embed=conf.model.params.n_embed,
            ckpt_path=f"/admin/home-mayavarma/vae-weights/{cfg.resume_from_ckpt}.ckpt",
        ).cuda()
        model.requires_grad_(False)
    elif mode == "ours":
        conf = OmegaConf.load(f"/admin/home-mayavarma/vae-weights/{cfg.resume_from_ckpt}.yaml")
        model = AutoencoderKL(
            ddconfig=conf.ddconfig,
            embed_dim=conf.embed_dim,
            ckpt_path=f"/admin/home-mayavarma/vae-weights/{cfg.resume_from_ckpt}.ckpt",
        ).cuda()
        model.requires_grad_(False)
    elif mode in ["bicubic", "bilinear", "nearest"]:
        cdim = cfg.img_size // int(cfg.resume_from_ckpt.split("-")[1][:-1])
        interp_modes = {"bicubic": 3, "bilinear": 2, "nearest": 0}
        i = interp_modes[mode]

        encode_transform = Resize((cdim, cdim), interpolation=i, antialias=True)
        decode_transform = Resize((cfg.img_size, cfg.img_size), interpolation=i, antialias=True)

    # Create reconstruction metrics
    names, metrics = list(zip(*cfg["metrics"]))
    metrics = list(zip(names, metrics))
    for _, metric in metrics:
        metric.cuda().reset()

    with torch.no_grad():
        num_batches = int(NUM_SAMPLES / cfg.dataloader.valid.batch_size)
        total_images = 0
        for step, batch in enumerate(valid_dataloader):
            print(f"=> Step: {step}/{num_batches}")
            if step == num_batches:
                break

            img = batch["img"].cuda()

            if mode == "ours" or mode == "sd_kl":
                rec, _ = model(img, sample_posterior=False)
            elif mode == "sd_vq":
                rec, _ = model(img)
            elif mode in ["bicubic", "bilinear", "nearest"]:
                rec = decode_transform(encode_transform(img)).cuda()

            if cfg.fine_grained:
                box = get_bbox(dataset_cat, cfg.img_size, batch["txt"])
                img = torchvision.transforms.functional.crop(img, box[0], box[1], box[2], box[3])
                rec = torchvision.transforms.functional.crop(rec, box[0], box[1], box[2], box[3])
            if mode == "sd_kl" or mode == "sd_vq":
                img = torch.mean(img, 1, keepdim=True)
                rec = torch.mean(rec, 1, keepdim=True)
            for _, metric in metrics:
                metric.update(img, rec)
            total_images += img.shape[0]

    rec_metrics = {}
    for name, metric in metrics:
        rec_metrics[name] = metric.compute().item()
    print(f"=> Computed reconstruction metrics for a sample of {total_images} images.")
    rec_metrics["Samples"] = total_images
    print(rec_metrics)

    out_path = f"metrics_rec/{cfg.img_size}_{cfg.resume_from_ckpt}_{dataset_cat}_{cfg.seed}.json"
    with open(out_path, "w") as f:
        json.dump(rec_metrics, f)


if __name__ == "__main__":
    main()
