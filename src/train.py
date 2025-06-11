import random
from dataclasses import dataclass
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
from .scaler import ArcTan, Cache, MinMax, Robust
from .utils import seq_split

torch.autograd.set_detect_anomaly(True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device.upper())


@dataclass
class TaskConfig:
    seq_len = 1024
    pred_len = 128
    enc_in = 3
    c_out = 3


@dataclass
class DataConfig:
    symbols = []
    timerange = "15m"
    val_size = 4000
    scaler_name = "Robust"


@dataclass
class TrainingConfig:
    learning_rate = 3e-3
    weight_decay = 1e-2
    gradient_clip = 1.0


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


def create_loaders(
    config: DataConfig, task: TaskConfig
) -> tuple[DataLoader, DataLoader]:
    datasets = []
    val_datasets = []
    for symbol in config.symbols:
        df = pd.read_pickle(f"data/{symbol}:USDT-{config.timerange}.pkl")
        dataset = TradeDataset(df, scaler_, task.seq_len, task.pred_len)
        train, val = seq_split(dataset, [len(dataset) - val_size, val_size])
        datasets.append(train)
        val_datasets.append(val)

    train_loader = DataLoader(
        ConcatDataset(datasets),
        batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=device == "cuda",
        persistent_workers=True,
    )
    val_loader = DataLoader(
        ConcatDataset(val_datasets),
        batch_size,
        num_workers=12,
        pin_memory=device == "cuda",
        persistent_workers=True,
    )
    return train_loader, val_loader


set_seeds(2003)
init_device(device)

data = DataConfig()
data.symbols = ["BTC", "ETH", "XRP"]

task = TaskConfig()
task.seq_len = 1024
task.pred_len = 128


batch_size = 512
epochs = 10
learn = 3e-3
val_size = 4000

scalers = {"Robust": Robust, "MinMax": MinMax, "ArcTan": ArcTan, "Cache": Cache}
scalern = "Robust"
scaler_ = scalers[scalern]()


config = DUETConfig()
config.seq_len = task.seq_len
config.pred_len = task.pred_len
config.enc_in = task.enc_in
config.c_out = task.c_out
model = DUET(config).to(device)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params: {params:,}")


criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learn, weight_decay=1e-2)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learn,
    epochs=epochs,
    steps_per_epoch=1000,
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

    for x, y, mark, context in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mark = mark.to(device, non_blocking=True)
        with autocast(device):
            outputs, _ = model(x, mark)

        # for i in range(len(outputs)):
        #     ctx = context[i].cpu().numpy()
        #     pred = outputs[i].cpu().numpy()
        #     target = y[i].cpu().numpy()

        #     pred = dataset.inverse(pred, ctx)
        #     target = dataset.inverse(target, ctx)

        # yield abs(pred - target).mean()
        yield abs(y - outputs).mean()


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
            exit()

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)

        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN in gradient of {name}")

        scaler.update()
        scheduler.step()

        yield loss.item()


train_loader, val_loader = create_loaders(data, task)

for epoch in range(0, epochs):
    index = 0
    pbar = tqdm(train(model, train_loader), desc="Training", unit="batch", total=len(train_loader))
    for loss in pbar:
        writer.add_scalar("Epoch/Loss", loss, index)
        pbar.set_postfix(loss=f"{loss:.4f}")
        index += 1

    index = 0
    pbar = tqdm(evaluate(model, val_loader), desc="Evalute", unit="batch", total=len(val_loader))
    for loss in pbar:
        writer.add_scalar("Epoch/RLss", loss, index)
        pbar.set_postfix(loss=f"{loss:.4f}")
        index += 1

    # print(
    #     f"Epoch {epoch + 1:03d}/{epochs} | Loss {avgloss:.5f} | RLss {avgrlss:.5f}",
    #     end="",
    # )

    # if avgrlss < best_loss:
    #     best_loss = avgrlss
    #     print("  *", end="")
    # print()

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
