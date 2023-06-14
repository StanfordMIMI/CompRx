import os
import warnings
from glob import glob
from typing import Any, Dict, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pydicom
import voxel as vx
import zarr
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

__all__ = [
    "get_metadata",
    "to_dataframe",
    "compute_mask",
    "apply_mask",
    "process_row",
    "process_image",
    "quantize_image",
    "get_store_info",
    "plot_stores",
]


def get_metadata(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """Get the metadata of a DICOM file.

    Args:
        path (Union[str, os.PathLike]): Path to the DICOM file.

    Returns:
        Dict[str, Any]: Dictionary containing the metadata.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dicom = pydicom.dcmread(path, stop_before_pixels=True)

    flat_dicom = {}
    for tag in dicom.file_meta:
        if tag.VR not in ["SQ", "OB", "OW", "OF", "SQ", "UN", "UT"]:
            flat_dicom[tag.keyword] = tag.value

    for tag in dicom:
        if tag.VR not in ["SQ", "OB", "OW", "OF", "SQ", "UN", "UT", "PN"]:
            value = tag.value

            if isinstance(value, pydicom.multival.MultiValue):
                for i, v in enumerate(value):
                    keyword = f"{tag.keyword}_{i}" if i > 0 else tag.keyword
                    flat_dicom[keyword] = v
            else:
                flat_dicom[tag.keyword] = tag.value

    return flat_dicom


def to_dataframe(path: Union[str, os.PathLike], num_workers: int = 1) -> "pl.DataFrame":
    """Convert a directory of DICOM files to a dataframe.

    Args:
        path (Union[str, os.PathLike]): Path to the directory.
        num_workers (int, optional): Number of workers to use. Defaults to 1.

    Returns:
        "pl.DataFrame": Dataframe containing the flattened metadata of the DICOM files.
    """
    dicoms = glob(path, recursive=True)

    metadata = process_map(get_metadata, dicoms, max_workers=num_workers, chunksize=100)
    return pl.from_pandas(pd.DataFrame(metadata)).with_column(pl.Series("path", dicoms))


def compute_mask(x, thresholds: np.ndarray = []) -> np.ndarray:
    """Compute the mask of the breast outline.

    Args:
        x (np.ndarray): Image.
        thresholds (np.ndarray, optional): Thresholds to try. Defaults to [].

    Returns:
        np.ndarray: Mask of the breast outline.
    """
    # return the complete image if there are no thresholds (left to try)
    if len(thresholds) == 0:
        mask = (x > np.amin(x)).astype(np.uint8)
    else:
        # compute the mask
        threshold = thresholds[0]
        mask = (x > threshold).astype(np.uint8)

    # find the contours in the mask, and sort them by area
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in contours]
    pairs = sorted(zip(contours, areas), key=lambda x: x[1], reverse=True)

    # sanity check: the largest contour has to be at least 3 times larger than the second largest
    if len(pairs) > 1 and pairs[0][1] < 3 * pairs[1][1] and len(thresholds) > 0:
        return compute_mask(x, thresholds[1:])

    # select the largest contour, and generate its bounding box
    contours = [c[0] for c in pairs]
    contours = [contours[0]]
    _, _, w, h = cv2.boundingRect(contours[0])

    # sanity check: make sure the contour is not too large
    if w > x.shape[1] * 0.99 and len(thresholds) > 0:
        return compute_mask(x, thresholds[1:])

    # sanity check: make sure the contour is not too small
    if w < 100 or h < 100:
        return np.ones_like(x, dtype=np.uint8)

    # create the mask with the largest contour
    mask = np.zeros_like(mask, dtype=np.uint8)
    mask = cv2.fillPoly(mask, contours, 1)
    return mask


def apply_mask(volume: "vx.MedicalVolume", mask: np.ndarray) -> "vx.MedicalVolume":
    """Apply a mask to a volume.

    Args:
        volume (vx.MedicalVolume): A Voxel MedicalVolume to apply the mask on.
        mask (np.ndarray): The outline of the anatomical structure of interest.

    Returns:
        vx.MedicalVolume: The masked volume.
    """
    volume *= mask[..., None]
    x, y, w, h = cv2.boundingRect(mask)

    volume.set_metadata("Rows", h)
    volume.set_metadata("Columns", w)
    return volume[y : y + h, x : x + w, ...]  # noqa: E203


def process_image(row: Dict[str, Any], trim_horizontal: bool = True) -> "vx.MedicalVolume":
    """Convert a DICOM file into a Voxel MedicalVolume.

    Applies the modality LUT, rescaling factors, windowing, and grayscale transformation. Chooses
    the

    Args:
        row (Dict[str, Any]): A row of the dataframe.

    Returns:
        vx.MedicalVolume: The processed image.
    """
    # Load preprocess
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            x = vx.load(row["path"])
        except:
            dcm = pydicom.dcmread(row["path"], force=True)
            arr = dcm.pixel_array
            if arr.ndim == 2:
                arr = arr[..., np.newaxis]

            del dcm.PixelData
            x = vx.MedicalVolume(arr, affine=np.eye(4), headers=[dcm])

        if trim_horizontal:
            x = x[27:-27, ...]

        x = (
            x.astype(np.float32)
            .apply_modality_lut(inplace=True)
            .apply_rescale(inplace=True, sync=True)
        )

    # Window, as specified in the metadata
    wc = row["WindowCenter"]
    ww = row["WindowWidth"]

    # Dynamic range
    min_pixel_value, max_pixel_value = np.amin(x), np.amax(x)
    new_ww = max_pixel_value - min_pixel_value
    new_wc = new_ww / 2 + min_pixel_value
    lb, ub = new_wc - new_ww / 2, new_wc + new_ww / 2

    if wc is None or ww is None or wc <= lb or wc >= ub:
        wc, ww = new_wc, new_ww
        x = x.apply_window(
            center=new_wc,
            width=new_ww,
            mode="LINEAR_EXACT",
            inplace=True,
            sync=True,
            output_range=(0, 1),
        ).to_grayscale(inplace=True, sync=True)
    else:
        x = x.apply_window(inplace=True, sync=True, output_range=(0, 1)).to_grayscale(
            inplace=True, sync=True
        )

    return x, wc, ww


def quantize_image(x: "vx.MedicalVolume", dynamic_range: int, dtype: np.integer) -> np.ndarray:
    """Quantize an image to a given dynamic range.

    Args:
        x (vx.MedicalVolume): The image to quantize.
        dynamic_range (int): The dynamic range to quantize to.
        dtype (np.integer): The data type to quantize to.

    Returns:
        np.ndarray: The quantized image.
    """
    # Scale to dynamic range
    x *= dynamic_range
    x = x.astype(dtype)
    return x


def process_row(
    row: Dict[str, Any],
    data_dir: Union[str, os.PathLike] = None,
    overwrite: bool = False,
    dtype: np.unsignedinteger = np.uint16,
    dynamic_range: int = None,
    plot: plt.Axes = None,
    debug: bool = False,
    use_mask: bool = True,
    shape: Tuple[slice, slice, slice] = None,
    trim_horizontal: bool = True,
):
    """Process one row of the dataframe.

    Args:
        row (Dict[str, Any]): A row of the dataframe.
        data_dir (Union[str, os.PathLike], optional): The directory to save the processed images to.
            Defaults to None.
        overwrite (bool, optional): Whether to overwrite existing images. Defaults to False.
        dtype (np.unsignedinteger, optional): The data type to quantize to. Defaults to np.uint16.
        dynamic_range (int, optional): The dynamic range to quantize to. Defaults to None.
        plot (plt.Axes, optional): The axes to plot the image to. Defaults to None.
        debug (bool, optional): Whether to print debug information. Defaults to False.
        use_mask (bool, optional): Whether to use the mask. Defaults to True.
        shape (Tuple[slice, slice, slice], optional): The shape to crop the image to (explicitly).
    """
    if data_dir is not None:
        output_path = os.path.join(data_dir, row["image_uuid"])
        if os.path.exists(output_path) and not overwrite:
            return None

    try:
        # Convert a row into a preprocessed MedicalVolume
        x, wc, ww = process_image(row, trim_horizontal=trim_horizontal)

        # Quantize to int to reduce storage size
        dynamic_range = dynamic_range if dynamic_range is not None else ww
        x = quantize_image(x, dynamic_range=dynamic_range, dtype=dtype)

        # Get mask
        thresholds = np.unique(x.volume)
        thresholds.sort()

        # Limit the maximum recursion depth to 100
        if len(thresholds) > 100:
            thresholds = thresholds[:100]

        # Compute the mask and plot if required
        mask = compute_mask(x.volume[..., 0], thresholds=thresholds)
        if plot is not None:
            plot.imshow(x.volume[..., 0], cmap="gray")
            plot.imshow(mask, alpha=0.5, cmap="jet")

        # Return results for debugging
        if debug:
            return x, mask, wc, ww

        # Store as Zarr file
        if data_dir is not None:
            if use_mask:
                x = apply_mask(x, mask)

            if shape is not None:
                x = x[shape]

            store = zarr.DirectoryStore(output_path)
            arr = x.to_zarr(
                store=store,
                read_only=False,
                mode="w",
                chunks=(384, 384, 1),
                affine_attr="affine",
            )

            # Set the metadata of the Zarr-file
            pixel_spacing = row.get("PixelSpacing", row.get("ImagerPixelSpacing", None))
            arr.attrs["patient_uuid"] = row["patient_uuid"]
            arr.attrs["study_uuid"] = row["study_uuid"]
            arr.attrs["series_uuid"] = row["series_uuid"]
            arr.attrs["image_uuid"] = row["image_uuid"]
            arr.attrs["dynamic_range"] = ww
            arr.attrs["window_center"] = wc
            arr.attrs["window_width"] = ww
            arr.attrs["pixel_spacing"] = (
                float(pixel_spacing) if pixel_spacing is not None else None
            )
            arr.attrs["photometric_interpretation"] = row["PhotometricInterpretation"]
            arr.attrs["bits_stored"] = int(row["BitsStored"])

    except Exception as e:
        print(f"Error processing {row['path']}")
        print(e)
        # raise e

        return None


def get_store_info(path: Union[str, os.PathLike]) -> Tuple[Tuple[int, int, int], float]:
    """Get the shape and storage ratio of a Zarr store.

    Args:
        path (Union[str, os.PathLike]): The path to the Zarr store.

    Returns:
        Tuple[Tuple[int, int, int], float]: The shape and storage ratio of the Zarr store.
    """
    with zarr.DirectoryStore(path) as store:
        arr = zarr.open_array(store=store)
        shape = arr.shape
        ratio = float(arr.info_items()[-2][1])
        return shape, ratio


def plot_stores(path: Union[str, os.PathLike], **kwargs):
    """Plot the distribution of the rows, columns and storage ratios.

    Args:
        path (Union[str, os.PathLike]): The path to the Zarr stores.
    """
    # Retrieve info about the Zarr stores
    info = process_map(get_store_info, glob(path), **kwargs)
    shapes, ratios = zip(*info)
    shapes = np.array(shapes)
    ratios = np.array(ratios)

    # Plot the distributions
    _, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].hist(shapes[:, 0], bins=500)
    axs[0].set_title("Rows")

    axs[1].hist(shapes[:, 1], bins=500)
    axs[1].set_title("Columns")

    axs[2].hist(ratios, bins=100)
    axs[2].set_title("Storage ratio")


def add_shapes(df: pl.DataFrame, root: Union[str, os.PathLike]) -> pl.DataFrame:
    """Add shape and storage ratio columns to the dataframe.

    Args:
        df (pl.DataFrame): The dataframe to add the columns to.

    Returns:
        pl.DataFrame: The dataframe with the added columns.
    """
    shapes, ratios = [], []
    for row in tqdm(df.rows(named=True)):
        shape, ratio = get_store_info(os.path.join(root, row["image_uuid"]))
        shapes.append(shape)
        ratios.append(ratio)

    shapes = np.array(shapes)
    shapes = pl.DataFrame(shapes, schema=["height", "width", "channels"]).with_columns(
        pl.Series(ratios).alias("ratio")
    )

    return pl.concat([df, shapes], how="horizontal")
