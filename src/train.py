import random
from pathlib import Path

import numpy as np
import torch
from torch import GradScaler, autocast, nn, optim
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from duet.config import DUETConfig
from duet.model import DUETModel

from .dataset import TradeDataset

torch.manual_seed(2003)
np.random.seed(2003)
random.seed(2003)


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using ", device)


writer = SummaryWriter(log_dir="runs/DUET ETH")


symbols = [
    "ETH",
    # "XRP",
    # "BNB",
    # "SOL",
    # "MOVR",
    # "ZRX",
    # "PEOPLE",
    # "WLD",
]
timerange = "5m"

seq_len = 256
pred_len = 48

batch_size = 386
epochs = 50
learn = 3e-3
val_size = 1000

checkpoint = Path("state") / f"S{seq_len}P{pred_len}:{timerange}.pth"

datasets = [
    TradeDataset(f"data/{symbol}:USDT-{timerange}.pkl", seq_len, pred_len)
    for symbol in symbols
]
dataset = ConcatDataset(datasets)
train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])

dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=12,
    pin_memory=True,
    shuffle=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=12,
    pin_memory=True,
    shuffle=True,
    persistent_workers=True,
)

config = DUETConfig()
config.seq_len = seq_len
config.pred_len = pred_len
config.enc_in = 4
model = DUETModel(config)
model.to(device)

criterion = nn.HuberLoss(reduction="none")
optimizer = optim.AdamW(model.parameters(), lr=learn)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learn,
    epochs=epochs,
    steps_per_epoch=len(dataloader),
    pct_start=0.3,
    anneal_strategy="cos",
)
scaler = GradScaler(device)
best_loss = float("inf")

weight = torch.tensor([2.8, 0.4, 0.4, 0.4], device=device)

if checkpoint.is_file():
    cpdata = torch.load(checkpoint)
    model.load_state_dict(cpdata["model"])
    optimizer.load_state_dict(cpdata["optimizer"])
    # scheduler.load_state_dict(cpdata["scheduler"])
    scaler.load_state_dict(cpdata["scaler"])
    # best_loss = cpdata["loss"]
    print(f"Loaded {best_loss}")


for epoch in range(0, epochs):
    model.train()
    losses = []

    batch_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}/{epochs}",
        leave=False,
        position=1,
        total=len(dataloader),
        unit="batch",
    )
    batch_bar.set_postfix(loss=f"{0:.4f}")

    for i, (x, y) in enumerate(batch_bar):
        optimizer.zero_grad()

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(device):
            outputs, _ = model(x)
            loss = criterion(outputs, y) * weight
            loss = loss.mean()

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        losses.append(loss.item())
        batch_bar.set_postfix(loss=f"{sum(losses) / len(losses):.4f}")

    avg_loss = sum(losses) / len(losses)
    min_loss = min(losses)
    max_loss = max(losses)

    model.eval()
    val_losses = []

    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(device), torch.no_grad():
            outputs, _ = model(x)
            loss = criterion(outputs, y) * weight
            loss = loss.mean()

        val_losses.append(loss.item())

    avg_val_loss = sum(val_losses) / len(val_losses)
    min_val_loss = min(val_losses)
    max_val_loss = max(val_losses)

    print(
        f"Epoch {epoch + 1:03d} | "
        f"Loss {max_loss:.5f}/{avg_loss:.5f}/{min_loss:.5f} | "
        f"RLss {max_val_loss:.5f}/{avg_val_loss:.5f}/{min_val_loss:.5f} | "
        f"LR {optimizer.param_groups[0]['lr']:.6f}",
        end="",
    )

    writer.add_scalar("Epoch/Loss", avg_loss, epoch)
    writer.add_scalar("Epoch/RLss", avg_val_loss, epoch)

    if avg_loss < best_loss:
        best_loss = avg_loss
        print("  *", end="")
    print()

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "loss": best_loss,
        },
        str(checkpoint),
    )

writer.close()
