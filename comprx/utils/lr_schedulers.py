import math

from torch.optim.lr_scheduler import _LRScheduler

__all__ = ["CosineScheduler"]


class CosineScheduler(_LRScheduler):
    def __init__(self, optimizer, max_epochs: int, verbose: bool = False):
        self.optimizer = optimizer
        self.max_epochs = max_epochs

        # last epoch should always be -1, use load_state_dict to resume
        super().__init__(optimizer, -1, verbose)

    def _compute_lr(self, param_group):
        init_lr = param_group["initial_lr"]
        current_lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * self.last_epoch / self.max_epochs))

        if "fixed_lr" in param_group and param_group["fixed_lr"]:
            return init_lr
        else:
            return current_lr

    def get_lr(self):
        return [self._compute_lr(param_group) for param_group in self.optimizer.param_groups]