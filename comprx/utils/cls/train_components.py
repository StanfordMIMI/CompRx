from collections import defaultdict
from time import time
from typing import Any, List, Tuple

import torch
from accelerate import Accelerator
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm import tqdm

from comprx.utils.extras import get_weight_dtype
from comprx.utils.transforms import to_dict

__all__ = ["training_epoch", "validation_epoch"]


def training_epoch(
    cfg: Any,
    epoch: int,
    accelerator: Accelerator,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    default_metrics: List[Metric],
    classif_metrics: List[Tuple[str, Metric]],
):
    """Train one epoch for a classifier."""

    # Setup metrics
    for metric in default_metrics:
        metric.reset()

    for _, metric in classif_metrics:
        metric.reset()

    metric_loss, metric_data, metric_batch = default_metrics
    model.train()
    batch_start = time()
    dtype = get_weight_dtype(accelerator)

    for i, batch in enumerate(dataloader):
        data_time = time() - batch_start

        with accelerator.accumulate(model):
            batch = to_dict(batch)
            images, targets = batch["img"].to(dtype), batch["lbl"]

            # Compute output and loss
            output = model(images)
            loss = criterion(output, targets)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

        batch_time = time() - batch_start

        # Update metrics
        metric_loss.update(loss)
        metric_data.update(data_time)
        metric_batch.update(batch_time)

        output, targets = accelerator.gather_for_metrics((output, targets))
        for _, metric in classif_metrics:
            metric.update(output, targets)

        # Logging
        print(
            f"\r[Epoch <{epoch:03}/{cfg['max_epoch']}>: Step <{i:03}/{len(dataloader)}>] - "
            + f"Data(s): {data_time:.3f} ({metric_data.compute():.3f}) - "
            + f"Batch(s): {batch_time:.3f} ({metric_batch.compute():.3f}) - "
            + f"Loss: {loss.item():.4f} ({metric_loss.compute():.4f}) \n"
        )

        if cfg.get("logger", False) and i % cfg["log_every_n_steps"] == 0:
            log_data = {
                "epoch": epoch,
                "mean_loss": metric_loss.compute(),
                "mean_data": metric_data.compute(),
                "mean_batch": metric_batch.compute(),
                "step": i,
                "step_global": len(dataloader) * epoch + i,
                "step_loss": loss,
                "step_data": data_time,
                "step_batch": batch_time,
            }

            for name, metric in classif_metrics:
                log_data[name] = metric.compute()

            for idx, pg in enumerate(optimizer.param_groups):
                name = pg["name"] if "name" in pg else f"param_group_{idx}"
                if "lr" in pg:
                    log_data[f"{name}/lr"] = pg["lr"]

                if "momentum" in pg:
                    log_data[f"{name}/momentum"] = pg["momentum"]

                if "weight_decay" in pg:
                    log_data[f"{name}/weight_decay"] = pg["weight_decay"]

            accelerator.log(log_data)

        batch_start = time()
        if cfg.get("fast_dev_run", False):
            break


def validation_epoch(
    cfg: Any,
    epoch: int,
    accelerator: Accelerator,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    default_metrics: List[Metric],
    classif_metrics: List[Tuple[str, Metric]],
    slice_metrics: List[Tuple[str, int, Metric]],
):
    """Validate one epoch for a classifier."""

    # setup metrics
    for metric in default_metrics:
        metric.reset()

    for _, metric in classif_metrics:
        metric.reset()

    slice_metric_counter = defaultdict(int)
    for _, _, metric in slice_metrics:
        metric.reset()

    metric_loss = default_metrics[0]
    metric_loss.reset()

    model.eval()
    dtype = get_weight_dtype(accelerator)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            try:
                batch = to_dict(batch)
                images, targets, group_id = (
                    batch["img"].to(dtype),
                    batch["lbl"],
                    batch["group_id"],
                )

                output = model(images)
                loss = criterion(output, targets)

                # Update metrics
                metric_loss.update(loss)

                output, targets, group_id = accelerator.gather_for_metrics(
                    (output, targets, group_id)
                )
                for _, metric in classif_metrics:
                    metric.update(output, targets)

                for n, g, metric in slice_metrics:
                    idx = group_id == g
                    if idx.sum() == 0:
                        continue
                    metric.update(output[idx], targets[idx])
                    slice_metric_counter[n] += idx.sum()

            except Exception as e:
                print("<= Exception in cls_eval =>")
                print(e)

            if cfg.get("fast_dev_run", False):
                return 0

        # logging
        print(f"Validation - loss: {loss.item():.4f} ({metric_loss.compute():.4f}) \n")
        if cfg.get("logger", False):
            log_data = {
                "valid/epoch": epoch,
                "valid/mean_loss": metric_loss.compute(),
                "valid/step_loss": loss,
            }

            for name, metric in classif_metrics:
                log_data[f"valid/{name}"] = metric.compute()

            for name, g, metric in slice_metrics:
                if slice_metric_counter[name] > 0:
                    log_data[f"valid_slice/{name}"] = metric.compute()

            accelerator.log(log_data)

    return metric_loss.compute()
