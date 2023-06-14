from typing import List

from torch.utils.data import ConcatDataset

__all__ = ["ConcatDataset"]


class ConcatDataset(ConcatDataset):
    def __init__(self, datasets, dataset_ids: List[int] = None):
        for i in range(len(datasets)):
            if callable(datasets[i]):
                datasets[i] = datasets[i]()

        if dataset_ids is not None:
            datasets = [ds for ds in datasets if ds.dataset_id in dataset_ids]

        super().__init__(datasets)
