import random
from pathlib import Path

import numpy as np
import torch
from torch import GradScaler, autocast, nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from trader.model import TradeModel

from .tdataset import TimeSeries
from .utils import seq_split

torch.manual_seed(2003)
np.random.seed(2003)
random.seed(2003)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

if device == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


seq_len = 256
hidden = 64

batch_size = 128
epochs = 50
learn = 3e-4
val_size = 5000


writer = SummaryWriter(log_dir="runs/TradeModel")

dataset = TimeSeries("eth-signal.pkl", seq_len)
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

model = TradeModel(3, 3, hidden).to(device)

criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1.0, 1.0], device=device))
optimizer = optim.Adam(model.parameters(), lr=learn)
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

checkpoint_path = Path("state") / "TM_S32.pth"
if checkpoint_path.is_file():
    cpdata = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(cpdata["model"])
    optimizer.load_state_dict(cpdata["optimizer"])
    # scheduler.load_state_dict(cpdata["scheduler"])
    scaler.load_state_dict(cpdata["scaler"])
    # best_loss = cpdata["loss"]
    print(f"Loaded {cpdata['loss']:.6f}")


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    losses = []
    for frame, x, y in loader:
        frame = frame.to(device, non_blocking=True)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(device):
            logits, _ = model(x, frame)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    return losses


for epoch in range(epochs):
    model.train()
    losses = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
    for frame, x, y in pbar:
        optimizer.zero_grad()

        frame = frame.to(device, non_blocking=True)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(device):
            logits, hc = model(x, frame)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        losses.append(loss.item())
        pbar.set_postfix(loss=f"{sum(losses) / len(losses):.6f}")

    eval_losses = evaluate(model, val_loader)

    avgloss = sum(losses) / len(losses)
    avgrlss = sum(eval_losses) / len(eval_losses)

    writer.add_scalar("Epoch/Loss", avgloss, epoch)
    writer.add_scalar("Epoch/RLss", avgrlss, epoch)

    print(
        f"Epoch {epoch + 1:03d} | Loss {avgloss:.6f} | RLss: {avgrlss:.6f}",
        end="",
    )

    if avgrlss < best_loss:
        best_loss = avgrlss
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
        str(checkpoint_path),
    )

writer.close()

val_loader._iterator._shutdown_workers()
dataloader._iterator._shutdown_workers()
