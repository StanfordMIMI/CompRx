from time import time
from typing import Any, Callable, Dict, List

from accelerate import Accelerator
from rich import print
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric

__all__ = ["training_epoch"]


def training_epoch(
    epoch: int,
    accelerator: Accelerator,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: Callable[[Dict[str, Tensor]], Tensor],
    optimizer: Optimizer,
    default_metrics: List[Metric],
    options: Dict[str, Any] = {},
):
    """Pretrain an SSL embedding for a single epoch."""

    for metric in default_metrics:
        metric.reset()

    metric_loss, metric_data, metric_batch = default_metrics
    model.train()

    epoch_start, batch_start = time(), time()
    for i, batch in enumerate(dataloader):
        # measure data loading time
        data_time = time() - batch_start

        # compute output and loss
        # batch_size = batch["img"][0].size(0)
        output = model(batch)

        loss = criterion(output)
        # losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        # gradient scaling for mixed precision
        accelerator.backward(loss)
        optimizer.step()

        # measure elapsed time
        # batch_time.update(time() - end)
        # end = time()

        batch_time = time() - batch_start
        metric_loss.update(loss.item())
        metric_data.update(data_time)
        metric_batch.update(batch_time)

        # Logging values
        print(
            f"\r[Epoch <{epoch:03}/{options.get('max_epoch', '?')}>: Step <{i:03}/{len(dataloader)}>] - "
            + f"Data(s): {data_time:.3f} ({metric_data.compute():.3f}) - "
            + f"Batch(s): {batch_time:.3f} ({metric_batch.compute():.3f}) - "
            + f"Loss: {loss.item():.3f} ({metric_loss.compute():.3f}) - "
            + f"{(((time() - epoch_start) / (i + 1)) * (len(dataloader) - i)) / 60:.2f} m remaining\n"
        )

        # step logging
        if options.get("is_logging", False) and i % options.get("log_every_n_steps", 1) == 0:
            log_data = {
                "epoch": epoch,
                "mean_loss": metric_loss.compute(),
                "mean_data": metric_data.compute(),
                "mean_batch": metric_batch.compute(),
                "step": i,
                "step_loss": loss.item(),
                "step_data": data_time,
                "step_batch": batch_time,
            }

            for idx, pg in enumerate(optimizer.param_groups):
                name = pg["name"] if "name" in pg else f"param_group_{idx}"
                log_data[f"{name}/momentum"] = pg.get("momentum", 0)
                log_data[f"{name}/weight_decay"] = pg.get("weight_decay", 0)
                log_data[f"{name}/lr"] = pg.get("lr", 0)

            accelerator.log(log_data)

        batch_start = time()

        if options.get("fast_dev_run", False):
            break
