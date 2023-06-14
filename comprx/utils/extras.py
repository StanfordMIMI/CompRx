import os
import random
from shutil import copyfile
from typing import Dict, List, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf
from tqdm.auto import tqdm

__all__ = ["sanitize_dataloader_kwargs"]


def sanitize_dataloader_kwargs(kwargs):
    """Converts num_workers argument to an int
    NB: this is needed if num_workers is gather from the OS environment.
    """

    if "num_workers" in kwargs:
        kwargs["num_workers"] = int(kwargs["num_workers"])

    return kwargs


def set_seed(seed: int):
    """Seed the RNGs."""

    print(f"=> Setting seed [seed={seed}]")
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True

    print("=> Setting a seed slows down training considerably!")


def get_weight_dtype(accelerator):
    """Get the weight dtype from the accelerator."""

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    return weight_dtype


def move_checkpoints(
    experiments: List[str],
    experiment_type: str,
    dataset: str,
    input_dir: Union[os.PathLike, str] = "/fsx/home-sluijs/comprx/logs/",
    output_dir: Union[os.PathLike, str] = "/fsx/aimi/comprx/checkpoints/",
):
    checkpoints = []
    for experiment in experiments:
        experiment_dir = os.path.join(input_dir, experiment)
        if os.path.isdir(experiment_dir):
            # runs
            run_dir = os.path.join(experiment_dir, "runs")
            if os.path.exists(run_dir):
                timestamps = sorted(os.listdir(run_dir))
                if len(timestamps) > 0:
                    timestamp = timestamps[-1]
                    timestamp_dir = os.path.join(run_dir, timestamp)
                    checkpoint_path = os.path.join(
                        timestamp_dir, "checkpoints", "last.pt", "pytorch_model.bin"
                    )
                    checkpoints.append((checkpoint_path, experiment))

            # multiruns
            multirun_dir = os.path.join(experiment_dir, "multiruns")
            if os.path.exists(multirun_dir):
                timestamps = sorted(os.listdir(multirun_dir))
                if len(timestamps) > 0:
                    timestamp = timestamps[-1]
                    timestamp_dir = os.path.join(multirun_dir, timestamp)
                    numbers = sorted(os.listdir(timestamp_dir))
                    if len(numbers) > 0:
                        number = numbers[-1]
                        number_dir = os.path.join(timestamp_dir, number)
                        checkpoint_path = os.path.join(
                            number_dir, "checkpoints", "last.pt", "pytorch_model.bin"
                        )
                        checkpoints.append((checkpoint_path, experiment))

    # move checkpoints to output directory
    for (checkpoint_path, experiment) in tqdm(checkpoints):
        if os.path.exists(checkpoint_path):
            output_path = os.path.join(
                output_dir, experiment_type, dataset, f"{experiment}" + "-last.pt"
            )
            copyfile(checkpoint_path, output_path)
