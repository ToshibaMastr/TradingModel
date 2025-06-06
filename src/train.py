import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import GradScaler, autocast, nn, optim
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from tsfs.models.duet import DUET, DUETConfig

from .dataset import TradeDataset
from .parseset import get_dafe
from .utils import seq_split

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using ", device)

seed = 2003

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if device == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


# LINK 0.4222
# DOGE 0.4865
# XLM 0.3860
# TRX 0.5600
# ADA 0.4227
# XRP 0.4735
# BNB 0.5465
# LTC 0.4220
# BTC 0.4141
# ETH 0.4642

symbols = [
    # "LINK",
    # "DOGE",
    # "XLM",
    # "TRX",
    #
    "ADA",
    "XRP",
    "BNB",
    #
    "LTC",
    "BTC",
    "ETH",
]
timerange = "15m"

seq_len = 1024 + 512
pred_len = 128

batch_size = 512
epochs = 50
learn = 3e-3
val_size = 4000

datasets = []
val_datasets = []
for symbol in symbols:
    df = pd.read_pickle(f"data/{symbol}:USDT-{timerange}.pkl")
    dataset = TradeDataset(df, seq_len, pred_len)
    train, val = seq_split(dataset, [len(dataset) - val_size, val_size])
    datasets.append(train)
    val_datasets.append(val)
train_dataset = ConcatDataset(datasets)
val_dataset = ConcatDataset(val_datasets)
dataloader = DataLoader(
    train_dataset,
    batch_size,
    num_workers=12,
    pin_memory=True,
    shuffle=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
)

config = DUETConfig()
config.seq_len = seq_len
config.pred_len = pred_len
config.enc_in = 4
config.c_out = 4
model = DUET(config).to(device)


params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params: {params:,}")

criterion = nn.SmoothL1Loss(beta=0.5)
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

total_epochs = 0
modelname = model.name
checkpoint = Path("state") / (modelname + ".pth")
if checkpoint.is_file():
    cpdata = torch.load(checkpoint, map_location=device)
    model.load_state_dict(cpdata["model"])
    optimizer.load_state_dict(cpdata["optimizer"])
    # scheduler.load_state_dict(cpdata["scheduler"])
    scaler.load_state_dict(cpdata["scaler"])
    best_loss = cpdata["loss"]
    total_epochs = cpdata["epochs"]
    print(f"Loaded {cpdata['loss']}")

log_dir = Path("runs")
writer = SummaryWriter(log_dir / modelname)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    losses = []

    batch_bar = tqdm(loader, desc="Real Loss", unit="batch")
    for x, y, mark in batch_bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mark = mark.to(device, non_blocking=True)
        with autocast(device):
            outputs, _ = model(x, mark)
            loss = criterion(outputs, y)
        losses.append(loss.item())
    return losses


get_dafe("data/ETH:USDT-15m.pkl", model, seq_len, pred_len, val_size, batch_size)
exit()

for epoch in range(0, epochs):
    model.train()
    losses = []

    batch_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
    for x, y, mark in batch_bar:
        optimizer.zero_grad()

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mark = mark.to(device, non_blocking=True)

        with autocast(device):
            outputs, _ = model(x, mark)
            loss = criterion(outputs, y)

        if not torch.isfinite(loss):
            print("null loss")
            exit()

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        losses.append(loss.item())
        batch_bar.set_postfix(loss=f"{sum(losses) / len(losses):.4f}")

    avgloss = sum(losses) / len(losses)
    writer.add_scalar("Epoch/Loss", avgloss, epoch)

    eval_losses = evaluate(model, val_loader)

    avgrlss = sum(eval_losses) / len(eval_losses)
    writer.add_scalar("Epoch/RLss", avgrlss, epoch)

    get_dafe("data/ETH:USDT-15m.pkl", model, seq_len, pred_len, val_size, batch_size)

    print(f"Epoch {epoch + 1:03d} | Loss {avgloss:.5f} | RLss {avgrlss:.5f}", end="")

    if avgrlss < best_loss:
        best_loss = avgrlss
        print("  *", end="")
    print()

    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epochs": total_epochs + epoch,
            "model": model.state_dict(),
            "loss": best_loss,
        },
        str(checkpoint),
    )


writer.close()

val_loader._iterator._shutdown_workers()
dataloader._iterator._shutdown_workers()
