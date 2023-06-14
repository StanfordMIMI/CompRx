from time import time
from typing import List, Tuple

import torch
from accelerate import Accelerator
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric

from comprx.utils.transforms import to_dict

__all__ = ["vae_cls_train", "vae_cls_eval"]


def vae_cls_train(
    epoch: int,
    accelerator: Accelerator,
    dataloader: DataLoader,
    vae: Tuple[nn.Module, nn.Module],
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    default_metrics: List[Metric],
    classif_metrics: List[Tuple[str, Metric]],
    is_logging: bool = False,
    log_every_n_steps: int = 1,
    fast_dev_run: bool = False,
    weight_dtype: torch.dtype = torch.float32,
):
    """Train one epoch for a classifier."""

    # setup metrics
    for metric in default_metrics:
        metric.reset()

    for _, metric in classif_metrics:
        metric.reset()

    metric_loss, metric_data, metric_batch = default_metrics
    model.train()
    batch_start = time()
    for i, batch in enumerate(dataloader):
        data_time = time() - batch_start

        with accelerator.accumulate(model):
            batch = to_dict(batch)
            images, targets = batch["img"].to(weight_dtype), batch["lbl"]
            encoder, quant_conv = vae
            out = encoder(images)
            out = quant_conv(out)
            latents = out[:, :4, ...]

            # compute output and loss
            output = model(latents)
            loss = criterion(output, targets)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

        batch_time = time() - batch_start

        # update metrics
        metric_loss.update(loss)
        metric_data.update(data_time)
        metric_batch.update(batch_time)

        output, targets = accelerator.gather_for_metrics((output, targets))
        for _, metric in classif_metrics:
            metric.update(output, targets)

        # logging
        print(
            f"[{epoch:03}: {i:03}/{len(dataloader)}] data: {data_time:.4f} ({metric_data.compute():.4f}) - batch: {batch_time:.4f} ({metric_batch.compute():.4f}) - loss: {loss.item():.4f} ({metric_loss.compute():.4f}) \n"
        )
        if is_logging and i % log_every_n_steps == 0:
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
        if fast_dev_run:
            break


def vae_cls_eval(
    epoch: int,
    accelerator: Accelerator,
    dataloader: DataLoader,
    vae: Tuple[nn.Module, nn.Module],
    model: nn.Module,
    criterion: nn.Module,
    default_metrics: List[Metric],
    classif_metrics: List[Tuple[str, Metric]],
    is_logging: bool = False,
    log_every_n_steps: int = 1,
    fast_dev_run: bool = False,
    weight_dtype: torch.dtype = torch.float32,
):
    """Validate one epoch for a classifier."""
    # setup metrics
    for metric in default_metrics:
        metric.reset()

    for _, metric in classif_metrics:
        metric.reset()

    metric_loss = default_metrics[0]
    metric_loss.reset()

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = to_dict(batch)
            images, targets = batch["img"].to(weight_dtype), batch["lbl"]
            encoder, quant_conv = vae
            out = encoder(images)
            out = quant_conv(out)
            latents = out[:, :4, ...]

            # compute output and loss
            output = model(latents)
            loss = criterion(output, targets)

            # update metrics
            metric_loss.update(loss)

            output, targets = accelerator.gather_for_metrics((output, targets))
            for _, metric in classif_metrics:
                metric.update(output, targets)

            # logging
            print(f"Validation - loss: {loss.item():.4f} ({metric_loss.compute():.4f}) \n")
            if is_logging and i % log_every_n_steps == 0:
                log_data = {
                    "valid/epoch": epoch,
                    "valid/mean_loss": metric_loss.compute(),
                    "valid/step_loss": loss,
                }

                for name, metric in classif_metrics:
                    log_data[f"valid/{name}"] = metric.compute()

                accelerator.log(log_data)

            if fast_dev_run:
                return 0

    return metric_loss.compute()
