import random
from pathlib import Path

import numpy as np
import torch
from torch import GradScaler, autocast, nn, optim
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from duet.config import DUETConfig
from duet.model import DUETModel

from .dataset import TradeDataset
from .utils import seq_split

torch.manual_seed(2003)
np.random.seed(2003)
random.seed(2003)


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using ", device)

if device == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

writer = SummaryWriter(log_dir="runs/DUET")


symbols = [
    # "ETH",
    # "ADA",
    # "XRP",
    # "BNB",
    "BTC",
    # "SOL",
    # "MOVR",
    # "ZRX",
    # "PEOPLE",
    # "WLD",
]
timerange = "5m"

seq_len = 1024
pred_len = 128

batch_size = 386
epochs = 25
learn = 3e-3
val_size = 1000

checkpoint = Path("state") / f"S{seq_len}P{pred_len}:{timerange}.pth"

datasets = [
    TradeDataset(f"data/{symbol}:USDT-{timerange}.pkl", seq_len, pred_len)
    for symbol in symbols
]
dataset = ConcatDataset(datasets)
train_dataset, val_dataset = seq_split(dataset, [len(dataset) - val_size, val_size])
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
model = DUETModel(config).to(device)

criterion = nn.MSELoss(reduction="none")
optimizer = optim.AdamW(model.parameters(), lr=learn)
scheduler = optim.lr_scheduler.OneCycleLR(
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
    cpdata = torch.load(checkpoint, map_location=device)
    model.load_state_dict(cpdata["model"])
    optimizer.load_state_dict(cpdata["optimizer"])
    # scheduler.load_state_dict(cpdata["scheduler"])
    scaler.load_state_dict(cpdata["scaler"])
    # best_loss = cpdata["loss"]
    print(f"Loaded {cpdata['loss']}")


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    losses = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(device):
            outputs, _ = model(x)
            loss = criterion(outputs[:, :, 0], y[:, :, 0])
            loss = loss.mean()
        losses.append(loss.item())
    return losses


for epoch in range(0, epochs):
    model.train()
    losses = []

    batch_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
    for x, y in batch_bar:
        optimizer.zero_grad()

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(device):
            outputs, _ = model(x)
            loss = criterion(outputs, y) * weight
            loss = loss.mean()

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        losses.append(loss.item())
        batch_bar.set_postfix(loss=f"{sum(losses) / len(losses):.4f}")

    for loss in losses:
        if not loss > 0:
            exit()

    for pair, loader in zip(symbols, datasets):
        eval_losses = evaluate(model, val_loader)
        print(f"{pair} | Loss {sum(eval_losses) / len(eval_losses):.5f}")

    avgloss = sum(losses) / len(losses)
    # avgrlss = sum(eval_losses) / len(eval_losses)

    writer.add_scalar("Epoch/Loss", avgloss, epoch)
    # writer.add_scalar("Epoch/RLss", avgrlss, epoch)

    print(
        f"Epoch {epoch + 1:03d} | Loss {avgloss:.5f}",
        end="",
    )

    if avgloss < best_loss:
        best_loss = avgloss
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

val_loader._iterator._shutdown_workers()
dataloader._iterator._shutdown_workers()
