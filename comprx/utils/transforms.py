import os
from typing import Any, Callable, Dict, List, Tuple, Union

import kornia.augmentation as K
import torch
import voxel as vx
import zarr

__all__ = ["to_dict", "load_store", "apply_namode", "ExtractKeys", "Expand", "RandomResizedCrop"]


def to_dict(x):
    if isinstance(x, dict):
        if "group_id" not in x:
            x["group_id"] = torch.zeros(
                (x["img"].size(0),), dtype=x["img"].dtype, device=x["img"].device
            )
        return x

    group = x[2] if len(x) == 3 else None
    return {"img": x[0], "lbl": x[1], "group_id": group}


def load_store(image_uuid: str, root: Union[str, os.PathLike], **kwargs):
    """Load a Zarr store from a given filename and root path."""

    path = os.path.join(root, image_uuid)
    store = zarr.DirectoryStore(path)

    kwargs = {"mode": "r", **kwargs}
    return vx.MedicalVolume.from_zarr(store=store, **kwargs)


def apply_namode(labels: torch.Tensor, na_mode: str = "positive") -> torch.Tensor:
    """Select the labels, apply the na_mode and return a Torch tensor."""
    assert na_mode is None or na_mode in ["positive", "negative"]

    if na_mode == "positive":
        return torch.where(
            labels == -1.0,
            torch.ones_like(labels),
            torch.where(labels == 1.0, labels, torch.zeros_like(labels)),
        )
    elif na_mode == "negative":
        return (torch.where(labels == 1.0, labels, torch.zeros_like(labels)),)

    return labels


class ExtractKeys:
    """Extract keys from a Dict sample."""

    def __init__(self, keys: List[str], unpack: bool = False):
        self.keys, self.unpack = keys, unpack

    def __call__(self, sample: Dict) -> Union[Dict, Any]:
        sample = {k: v for (k, v) in sample.items() if k in self.keys}

        if not self.unpack:
            return sample

        keys = list(sample.keys())
        if len(keys) > 1:
            raise ValueError("Can't unpack dicts with more than 1 key.")


class Expand:
    """Expand a tensor to the specified number of channels."""

    def __init__(self, num_channels: int):
        self.num_channels = num_channels

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze().expand(self.num_channels, -1, -1)


class RandomResizedCrop:
    def __init__(self, size, scale, **kwargs):
        kwargs = {"size": tuple(size), "scale": tuple(scale), **kwargs}
        self.fn = K.RandomResizedCrop(**kwargs)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


class ToSiamese:
    """Split augmentation pipeline into two branches."""

    def __init__(self, t1: Callable, t2: Callable):
        self.t1, self.t2 = t1, t2

    def __call__(self, x: Any) -> Tuple[Any, Any]:
        return [self.t1(x), self.t2(x)]
