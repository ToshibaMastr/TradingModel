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
from .scaler import ArcTan, Cache, MinMax, Robust
from .utils import seq_split

# torch.autograd.set_detect_anomaly(True)


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


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device.upper())

set_seeds(2003)
init_device(device)


symbols = [
    "BTC",
    "ETH",
    "XRP",
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
timerange = "15m"

seq_len = 1024
pred_len = 128

batch_size = 512
epochs = 10
learn = 3e-3
val_size = 4000

scalers = {"Robust": Robust, "MinMax": MinMax, "ArcTan": ArcTan, "Cache": Cache}

price_scaler = "Cache"
volume_scaler = "MinMax"

price_scaler_ = scalers[price_scaler]()
volume_scaler_ = scalers[volume_scaler]()

datasets = []
val_datasets = []
for symbol in symbols:
    df = pd.read_pickle(f"data/{symbol}:USDT-{timerange}.pkl")[-50000:]
    dataset = TradeDataset(df, price_scaler_, volume_scaler_, seq_len, pred_len)
    train, val = seq_split(dataset, [len(dataset) - val_size, val_size])
    datasets.append(train)
    val_datasets.append(val)

dataloader = DataLoader(
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

config = DUETConfig()
config.seq_len = seq_len
config.pred_len = pred_len
config.enc_in = 4
config.c_out = 4
model = DUET(config).to(device)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params: {params:,}")


criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learn, weight_decay=1e-2)
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
modelname = model.name + f"{seq_len}{price_scaler}{volume_scaler}"
checkpoint = Path("state") / (modelname + ".pth")
if checkpoint.is_file() and False:
    cpdata = torch.load(checkpoint, map_location=device, weights_only=False)
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

    batch_bar = tqdm(loader, desc="Evaluate", unit="batch")
    for x, y, mark, context in batch_bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mark = mark.to(device, non_blocking=True)
        with autocast(device):
            outputs, _ = model(x, mark)

        for b in range(len(outputs)):
            ctx = context[b].cpu().numpy()
            pred = outputs[b].cpu().numpy()
            target = y[b].cpu().numpy()

            pred = dataset.inverse(pred, ctx)
            target = dataset.inverse(target, ctx)

            pred = pred[:, 0:1]
            target = target[:, 0:1]

            # import plotly.graph_objects as go
            # fig = go.Figure()
            # fig.add_trace(go.Scatter(y=pred.flatten(), name="Predictions"))
            # fig.add_trace(go.Scatter(y=target.flatten(), name="Target"))
            # fig.show()
            # exit()

            losses.extend(abs(pred - target).mean(axis=1))
    return np.array(losses)


def train(model, loader):
    model.train()
    losses = []

    batch_bar = tqdm(loader, desc="Train", unit="batch")
    for x, y, mark, _ in batch_bar:
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

        losses.append(loss.item())
        batch_bar.set_postfix(loss=f"{sum(losses) / len(losses):.4f}")

    return np.array(losses)


for epoch in range(0, epochs):
    avgloss = train(model, dataloader).mean()
    writer.add_scalar("Epoch/Loss", avgloss, epoch)

    avgrlss = evaluate(model, val_loader).mean()
    writer.add_scalar("Epoch/RLss", avgrlss, epoch)

    # ataset_ = pd.read_pickle("data/ETH:USDT-15m.pkl")[-8000:]
    # et_dafe(
    #    dataset_,
    #    price_scaler_,
    #    volume_scaler_,
    #    model,
    #    seq_len,
    #    pred_len,
    #    val_size,
    #    batch_size,
    #

    # index = len(dataset_) - pred_len
    # dataset_["pred"] = np.nan
    # pred = predict(
    #     model,
    #     price_scaler_,
    #     volume_scaler_,
    #     dataset_,
    #     index,
    #     seq_len,
    #     pred_len,
    #     device=device,
    # )
    # dataset_.loc[pred.index, "pred"] = pred["close"]
    # show(dataset_[index - seq_len : index + pred_len])

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
            "epochs": total_epochs + epoch,
            "model": model.state_dict(),
            "loss": best_loss,
        },
        str(checkpoint),
    )

writer.close()

val_loader._iterator._shutdown_workers()
dataloader._iterator._shutdown_workers()
