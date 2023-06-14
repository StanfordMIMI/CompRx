import os
from typing import Callable, Union

import numpy as np
import polars as pl
import torch
import zarr

__all__ = [
    "ZarrLoader",
    "load_zarr",
    "load_labels",
    "load_tensor",
]


class ZarrLoader:
    """Zarr Loading Pipeline w/ Random Cropping.

    This is simple pipeline that support efficient random cropping without having to load the
    entire dataset into memory first.
    """

    def __init__(
        self,
        size: int = None,
        *,
        dtype: torch.dtype = torch.float32,
        padding: bool = True,
        store_type="directory",
    ):
        self.size = size
        self.dtype = dtype
        self.padding = padding
        self.store_type = store_type

    def __call__(self, x: Union[str, os.PathLike]):
        x = load_zarr(x, store_type=self.store_type)
        dynamic_range = x.attrs.get("dynamic_range", np.amax(x))

        if self.size is not None:
            h, w = x.shape[:2]

            if self.size >= h:
                i = slice(None)
            else:
                # Get random indices
                i = np.random.randint(0, h - self.size)
                i = slice(i, i + self.size)

            if self.size >= w:
                j = slice(None)
            else:
                j = np.random.randint(0, w - self.size)
                j = slice(j, j + self.size)

            # Return cropped image
            x = x[i, j]
            x = np.squeeze(x[:])

            # Pad cropped image
            if self.padding:
                pad_width = (0, max(self.size - h, 0)), (0, max(self.size - w, 0))
                x = np.pad(x, pad_width, "constant")
        else:
            x = x[:]

        if x.dtype == np.uint16:
            x = x.astype(np.float32)

        x = torch.from_numpy(x).squeeze().unsqueeze(0)

        return x.type(self.dtype) / dynamic_range


def load_labels(
    df: pl.DataFrame,
    dtype: np.dtype = None,
    # fill_null=None,
    fill_nan: float = None,
    squeeze: int = None,
) -> torch.Tensor:
    """Load the labels from a dataframe."""
    # BUG: Polars hangs when trying to convert to numpy in a DataLoader
    x = df.to_pandas().to_numpy()
    if dtype is not None:
        x = x.astype(dtype)

    if isinstance(squeeze, int):
        out = torch.from_numpy(x).squeeze(dim=squeeze)
    else:
        out = torch.from_numpy(x).squeeze()

    if isinstance(fill_nan, float):
        out = torch.where(out.isnan(), fill_nan, out)

    return out


def load_tensor(path: str, dtype: torch.dtype = torch.float32, **kwargs):
    """Auto load files based on file extensions."""
    ext = os.path.splitext(path)[-1]
    if ext == ".pt":
        x = torch.load(path, **kwargs)
        if len(x.shape) < 3:
            return x.squeeze().unsqueeze(0).type(dtype)
        else:
            return x.type(dtype)

    if ext in [".npy", ".npz"]:
        x = np.load(path, **kwargs)

        if ext == ".npz":
            x = x[x.files[0]]

        if x.dtype == np.uint16:
            x = x.astype(np.float32)

        return torch.from_numpy(x).squeeze().unsqueeze(0).type(dtype)

    raise NotImplementedError(f"Extension {ext} not supported.")


def load_txt(path: str):
    """Load a text file."""
    with open(path) as f:
        return f.read()


def load_zarr(
    path: str,
    *,
    store_constructor: Callable = None,
    to_tensor: bool = False,
    dtype: torch.dtype = None,
    **kwargs,
):
    """Load a Zarr Directory Store."""

    if store_constructor is not None:
        path = store_constructor(path, **kwargs)

    arr = zarr.open_array(store=path, mode="r")
    if dtype is not None:
        arr = arr[:].astype(dtype)

    if to_tensor:
        arr = torch.from_numpy(arr[:])

    return arr
