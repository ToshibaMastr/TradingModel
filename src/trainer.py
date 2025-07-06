from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import torch
from torch import GradScaler, nn, optim
from torch.optim import lr_scheduler


@dataclass
class TrainingConfig:
    epochs: int = 20

    learning_rate: float = 3e-3
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0


class Trainer:
    def __init__(
        self,
        params: Iterator[nn.Parameter],
        train_loader,
        config: TrainingConfig,
        device: str = "cuda",
    ):
        self.total_epochs = 0

        self.params = params
        self.config = config
        self.device = device

        self.optimizer = optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy="cos",
        )

        self.scaler = GradScaler(device)

    def epochs(self):
        while self.total_epochs < self.config.epochs:
            yield self.total_epochs
            self.total_epochs += 1

    def step(self, loss: torch.Tensor):
        # if not torch.isfinite(loss):
        #     print("⚠️ Null loss detected")

        self.scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(self.params, self.config.gradient_clip)

        # for name, param in self.model.named_parameters():
        #     if param.grad is not None and torch.isnan(param.grad).any():
        #         print(f"NaN in gradient of {name}")

        self.scaler.step(self.optimizer)

        setattr(self.optimizer, "_opt_called", True)  # fix scheduler wrapper

        self.scaler.update()
        self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epochs": self.total_epochs,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]):
        self.total_epochs = state_dict["epochs"]
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.scaler.load_state_dict(state_dict["scaler"])

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]
