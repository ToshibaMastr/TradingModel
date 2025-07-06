from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.progress import BarColumn, Progress, TimeRemainingColumn
from torch import autocast, nn
from torch.utils.tensorboard import SummaryWriter
from tsfs.core import TaskConfig
from tsfs.models.duet import DUET, DUETConfig

from .data import DatasetR
from .trainer import Trainer, TrainingConfig
from .utils import init_device, set_seeds, user_data_init

torch.autograd.set_detect_anomaly(False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device.upper())


user_data = Path("user_data")
runname = "sambucha"

set_seeds(2003)
init_device(device)
user_data_init(user_data)


symbols = [
    "BTC",
    # "ETH",
    # "XRP",
    # "BNB",
    # "SOL",
    # "HBAR",
    # "ADA",
    # "TRX",
    # "USDC",
    # "DOGE",
    # "SUI",
    # "LINK",
    # "AVAX",
    # "XLM",
    # "BCH",
    # "TON",
    # "LTC",
    # "DOT",
    # "XMR",
]

dataset = DatasetR(user_data / "data", symbols, "1m")
now = datetime.now()
dataset.download(100_000)
task = TaskConfig(1024, 32, enc_in=4, c_out=3)
train_loader, val_loader = dataset.loaders(task, [20000], device=device)

mconfig = DUETConfig()
mconfig.apply(task)
model = DUET(mconfig).to(device)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📊 Parameters: {params:,}")


tconfig = TrainingConfig()
trainer = Trainer(model.parameters(), train_loader, tconfig, device=device)


criterion = nn.MSELoss()
best_loss = float("inf")

total_epochs = 0
checkpoint = user_data / "state" / (runname + ".pth")
if checkpoint.is_file():
    cpdata = torch.load(checkpoint, map_location=device)
    model.load_state_dict(cpdata["model"])
    trainer.load_state_dict(cpdata["trainer"])
    best_loss = cpdata["loss"]
    print(f"✅ Loaded {best_loss}")


def train(model, loader):
    model.train()

    for x, y, mark, _ in loader:
        trainer.zero_grad()

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mark = mark.to(device, non_blocking=True)

        with autocast(device):
            outputs, _ = model(x, mark)
            loss = criterion(outputs, y)

        if not torch.isfinite(loss):
            print("⚠️ Null loss detected")
            dataset.cleanup()
            exit()

        trainer.step(loss)

        yield loss.item()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    for x, y, mark, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mark = mark.to(device, non_blocking=True)
        with autocast(device):
            outputs, _ = model(x, mark)
            loss = criterion(outputs, y)

        if not torch.isfinite(loss):
            print("⚠️ Null loss detected")
            dataset.cleanup()
            exit()

        yield loss.item()


writer = SummaryWriter(user_data / "runs" / runname)
console = Console()

for epoch in trainer.epochs():
    train_loss = np.empty([len(train_loader)])
    with Progress(
        "[bold blue]Training[/] • {task.completed}/{task.total}",
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        TimeRemainingColumn(),
        "•",
        "[yellow]{task.fields[avg_loss]:.4f}[/yellow]",
        transient=True,
    ) as progress:
        ptask = progress.add_task("training", total=len(train_loader), avg_loss=0)
        for i, loss in enumerate(train(model, train_loader)):
            train_loss[i] = loss
            running_loss = train_loss[: i + 1].mean()
            progress.update(ptask, advance=1, avg_loss=running_loss)
    avgloss = train_loss.mean()

    evaluate_loss = np.empty([len(val_loader)])
    with Progress(
        "[bold green]Evaluating[/] • {task.completed}/{task.total}",
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        TimeRemainingColumn(),
        "•",
        "[green]{task.fields[avg_loss]:.4f}[/green]",
        transient=True,
    ) as progress:
        ptask = progress.add_task("evaluating", total=len(val_loader), avg_loss=0)
        for i, loss in enumerate(evaluate(model, val_loader)):
            evaluate_loss[i] = loss
            running_loss = evaluate_loss[: i + 1].mean()
            progress.update(ptask, advance=1, avg_loss=running_loss)
    avgrlss = evaluate_loss.mean()

    writer.add_scalar("train/loss", avgloss, epoch)
    writer.add_scalar("train/real loss", avgrlss, epoch)

    console.print(
        f"Epoch {epoch + 1:03d}/{tconfig.epochs} | "
        f"Loss [yellow]{avgloss:.5f}[/yellow] | "
        f"RLss [green]{avgrlss:.5f}[/green]",
        end="",
    )

    if avgrlss < best_loss:
        best_loss = avgrlss
        console.print("  [bold red]*[/bold red]", end="")
    console.print()

    torch.save(
        {
            "trainer": trainer.state_dict(),
            "model": model.state_dict(),
            "loss": float(best_loss),
        },
        str(checkpoint),
    )

writer.close()
dataset.cleanup()
