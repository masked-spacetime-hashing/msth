from nerfstudio.engine.schedulers import Scheduler, SchedulerConfig
from dataclasses import dataclass, field
from typing import Optional, Type

import numpy as np
from torch.optim import Optimizer, lr_scheduler
from typing_extensions import Literal

from nerfstudio.configs.base_config import InstantiateConfig


@dataclass
class CosineDecaySchedulerConfig(SchedulerConfig):
    """Config for exponential decay scheduler with warmup"""

    _target: Type = field(default_factory=lambda: CosineDecayScheduler)
    """target class to instantiate"""
    lr_pre_warmup: float = 1e-8
    """Learning rate before warmup."""
    lr_final: Optional[float] = None
    """Final learning rate. If not provided, it will be set to the optimizers learning rate."""
    warmup_steps: int = 0
    """Number of warmup steps."""
    max_steps: int = 100000
    """The maximum number of steps."""
    ramp: Literal["linear", "cosine"] = "cosine"
    """The ramp function to use during the warmup."""


class CosineDecayScheduler(Scheduler):
    """Exponential decay scheduler with linear warmup. Scheduler first ramps up to `lr_init` in `warmup_steps`
    steps, then exponentially decays to `lr_final` in `max_steps` steps.
    """

    config: CosineDecaySchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> lr_scheduler._LRScheduler:
        if self.config.lr_final is None:
            lr_final = lr_init
        else:
            lr_final = self.config.lr_final

        def func(step):
            if step < self.config.warmup_steps:
                if self.config.ramp == "cosine":
                    lr = self.config.lr_pre_warmup + (1 - self.config.lr_pre_warmup) * np.sin(
                        0.5 * np.pi * np.clip(step / self.config.warmup_steps, 0, 1)
                    )
                else:
                    lr = (
                        self.config.lr_pre_warmup
                        + (lr_init - self.config.lr_pre_warmup) * step / self.config.warmup_steps
                    )
            else:
                t = np.clip(
                    (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps), 0, 1
                )
                # lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
                lr = lr_final + 0.5 * (lr_init - lr_final) * (1 + np.cos(np.pi * t))
            return lr / lr_init  # divided by lr_init because the multiplier is with the initial learning rate

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return scheduler
