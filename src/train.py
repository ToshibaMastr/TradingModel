from pathlib import Path

import torch
from torch import GradScaler, autocast, nn, optim
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from .dataset import TradeDataset
from .download import ExchangeDownloader
from .duet.config import DUETConfig
from .duet.model import DUETModel
from .parseset import predict, show

symbols = [
    # "XRP",
    # "BNB",
    # "SOL",
    "ADA",
    # "MOVR",
    # "ZRX",
    # "PEOPLE",
    # "WLD",
]
timerange = "5m"

seq_len = 256
pred_len = 48

batch_size = 256
epochs = 999
learn = 3e-3

checkpoint = Path("state") / f"S{seq_len}P{pred_len}:{timerange}.pth"

datasets = [
    TradeDataset(f"data/{symbol}:USDT-{timerange}.pkl", seq_len, pred_len, 400_000)
    for symbol in symbols
]
dataset = ConcatDataset(datasets)
dataloader = DataLoader(
    dataset, batch_size=batch_size, num_workers=12, pin_memory=True, shuffle=True
)

config = DUETConfig()
config.seq_len = seq_len
config.pred_len = pred_len
config.enc_in = 4
model = DUETModel(config)
# model = torch.compile(model)
model.to("cuda")

criterion = nn.MSELoss(reduction="none")
optimizer = optim.AdamW(model.parameters(), lr=learn)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learn,
    epochs=epochs,
    steps_per_epoch=len(dataloader),
    pct_start=0.3,
    anneal_strategy="cos",
)
scaler = GradScaler("cuda")
best_loss = float("inf")

weight = torch.tensor([2.8, 0.4, 0.4, 0.4], device="cuda")

if checkpoint.is_file():
    cpdata = torch.load(checkpoint)
    model.load_state_dict(cpdata["model"])
    optimizer.load_state_dict(cpdata["optimizer"])
    # scheduler.load_state_dict(cpdata["scheduler"])
    scaler.load_state_dict(cpdata["scaler"])
    # best_loss = cpdata["loss"]
    print(f"Loaded {best_loss}")

ed = ExchangeDownloader()
df = ed.download("ADA/USDT:USDT", timerange, 1000)
print(f"Test ADA/USDT:USDT {timerange} Downloaded")

for epoch in range(epochs):
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

        x = x.to("cuda", non_blocking=True)
        y = y.to("cuda", non_blocking=True)

        with autocast("cuda"):
            outputs, _ = model(x)
            loss = criterion(outputs, y) * weight
            loss = loss.mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss = loss.item()
        losses.append(loss)
        batch_bar.set_postfix(loss=f"{loss:.4f}")

    avg_loss = sum(losses) / len(losses)
    min_loss = min(losses)
    max_loss = max(losses)

    for loss in losses:
        if not loss > 0:
            exit()

    index = 1000 - seq_len - pred_len
    pred, lts = predict(model, df, index, seq_len, pred_len)
    print(
        f"Epoch {epoch + 1:03d} | "
        f"Loss {max_loss:.5f}/{avg_loss:.5f}/{min_loss:.5f} | "
        f"LTS {lts:.5f} | "
        f"LR {optimizer.param_groups[0]['lr']:.6f}",
        end="",
    )

    if avg_loss < best_loss:
        best_loss = avg_loss
        print("  *", end="")

        df.loc[pred.index, "pred"] = pred
        show(df)
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
