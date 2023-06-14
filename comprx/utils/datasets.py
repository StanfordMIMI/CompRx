from typing import List, Tuple

import numpy as np
import polars as pl
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm.auto import tqdm

__all__ = [
    "CHEXPERT_PATHOLOGIES",
    "VINDR_PATHOLOGIES",
    "compute_normalization_constants",
    "preprocess_chexpert_labels",
]


CHEXPERT_PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

VINDR_PATHOLOGIES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Clavicle fracture",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Enlarged PA",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Lung cavity",
    "Lung cyst",
    "Mediastinal shift",
    "Nodule/Mass",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
    "Rib fracture",
    "Other lesion",
    "COPD",
    "Lung tumor",
    "Pneumonia",
    "Tuberculosis",
    "Other diseases",
    "No finding"
]


def compute_normalization_constants(ds: Dataset, limit: int = None) -> Tuple[float, float]:
    """Compute the normalization constants for a dataset."""
    arr = []
    length = len(ds) if not limit else limit
    for idx in tqdm(range(length), disable=not ds.verbose):
        arr.append(np.ravel(ds[idx][0]))

    arr = np.concatenate(arr)
    return (np.mean(arr), np.std(arr))


def preprocess_chexpert_labels(df: pl.DataFrame, labels: List[str], na_mode: str = None) -> Tensor:
    """Select the labels, apply the na_mode and return a Torch tensor."""
    lbls = torch.from_numpy(df.select(labels).to_numpy())

    if na_mode == "positive":
        lbls = torch.where(
            lbls == -1.0,
            torch.ones_like(lbls),
            torch.where(lbls == 1.0, lbls, torch.zeros_like(lbls)),
        )
    elif na_mode == "negative":
        lbls = (torch.where(lbls == 1.0, lbls, torch.zeros_like(lbls)),)

    return lbls
