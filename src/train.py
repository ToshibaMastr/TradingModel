import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import GradScaler, autocast, nn, optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from tsfs.core import TaskConfig
from tsfs.models.duet import DUET, DUETConfig

from .data import DatasetR
from .scaler.robust import Robust

torch.autograd.set_detect_anomaly(False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device.upper())


@dataclass
class TrainingConfig:
    learning_rate: float = 3e-3
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0


def user_data_init(path: Path):
    for dir in ["runs", "state", "data"]:
        (user_data / dir).mkdir(parents=True, exist_ok=True)


def set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def init_device(device: str):
    if device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


user_data = Path("user_data")

set_seeds(2003)
init_device(device)
user_data_init(user_data)

runname = "sambucha"

symbols = [
    "BTC",
    # "ETH",
    # "XRP",
    # "BNB",
    # "SOL",
    # "USDC",
    # "DOGE",
    # "TRX",
    # "ADA",
    # "SUI",
    # "LINK",
    # "AVAX",
    # "XLM",
    # "BCH",
    # "TON",
    # "HBAR",
    # "LTC",
    # "DOT",
    # "XMR",
]

dataset = DatasetR(user_data / "data", symbols)
task = TaskConfig(1024, 32, enc_in=5, c_out=3)
training = TrainingConfig()

config = DUETConfig()
config.apply(task)

model = DUET(config).to(device)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params: {params:,}")


epochs = 60


criterion = nn.MSELoss()
optimizer = optim.AdamW(
    model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
)
scheduler = lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=training.learning_rate,
    epochs=epochs,
    steps_per_epoch=5875,
    pct_start=0.3,
    anneal_strategy="cos",
)
scaler = GradScaler(device)
best_loss = float("inf")

total_epochs = 0
checkpoint = user_data / "state" / (runname + ".pth")
if checkpoint.is_file():
    cpdata = torch.load(checkpoint, map_location=device)
    model.load_state_dict(cpdata["model"])
    optimizer.load_state_dict(cpdata["optimizer"])
    # scheduler.load_state_dict(cpdata["scheduler"])
    scaler.load_state_dict(cpdata["scaler"])
    best_loss = cpdata["loss"]
    total_epochs = cpdata["epochs"]
    print(f"Loaded {cpdata['loss']}")


@torch.no_grad()
def evaluate(model, loader):
    rb = Robust()
    model.eval()

    for x, y, mark, context in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mark = mark.to(device, non_blocking=True)
        with autocast(device):
            outputs, _ = model(x, mark)

        loss = torch.tensor(0.0, device=device)
        for pred, target, cont in zip(outputs, y, context[:, 0]):
            pred = rb.unscale(pred, cont)
            target = rb.unscale(target, cont)
            loss += abs(pred - target).mean()

        if not torch.isfinite(loss):
            print("null loss")
            # dataset.cleanup()
            continue  # exit()

        yield loss


def train(model, loader):
    model.train()

    for x, y, mark, _ in loader:
        optimizer.zero_grad()

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mark = mark.to(device, non_blocking=True)

        with autocast(device):
            outputs, _ = model(x, mark)
            loss = criterion(outputs, y)

        if not torch.isfinite(loss):
            print("null loss")
            # dataset.cleanup()
            continue  # exit()

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip)
        scaler.step(optimizer)

        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN in gradient of {name}")

        scaler.update()
        scheduler.step()

        yield loss.item()


writer = SummaryWriter(user_data / "runs" / runname)

train_loader, val_loader = dataset.loaders(task, [4000], device=device)

for epoch in range(total_epochs, epochs):
    train_loss = np.ndarray([len(train_loader)])
    pbar = tqdm(
        train(model, train_loader),
        desc="Training",
        unit="batch",
        total=len(train_loader),
    )
    for i, loss in enumerate(pbar):
        train_loss[i] = loss
        writer.add_scalar("train/loss", loss, i)
        pbar.set_postfix(loss=f"{train_loss[: i + 1].mean():.4f}")
    avgloss = train_loss.mean()

    evalute_loss = np.ndarray([len(val_loader)])
    pbar = tqdm(
        evaluate(model, val_loader), desc="Evalute", unit="batch", total=len(val_loader)
    )
    for i, loss in enumerate(pbar):
        evalute_loss[i] = loss
        writer.add_scalar("train/real loss", loss, i)
        pbar.set_postfix(loss=f"{evalute_loss[: i + 1].mean():.4f}")
    avgrlss = evalute_loss.mean()

    print(
        f"Epoch {epoch + 1:03d}/{epochs} | Loss {avgloss:.5f} | RLss {avgrlss:.5f}",
        end="",
    )

    if avgrlss < best_loss:
        best_loss = avgrlss
        print("  *", end="")
    print()

    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epochs": total_epochs + epoch + 1,
            "model": model.state_dict(),
            "loss": float(best_loss),
        },
        str(checkpoint),
    )

writer.close()
dataset.cleanup()
