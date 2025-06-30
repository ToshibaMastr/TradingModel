import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from tsfs.core import TaskConfig
from tsfs.models.duet import DUET, DUETConfig

from .data import ExchangeDownloader
from .data.dataset import TradeDataset
from .parseset import genf, get_dafe, predict, show

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device.upper())


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

runname = "sambucha"

task = TaskConfig(1024, 32)

config = DUETConfig()
config.apply(task)

model = DUET(config).to(device)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params: {params:,}")


checkpoint = user_data / "state" / (runname + ".pth")
if checkpoint.is_file():
    cpdata = torch.load(checkpoint, weights_only=False)
    model.load_state_dict(cpdata["model"])
    print(f"✅ Checkpoint m loaded. Loss {cpdata['loss']:.6f}")

# df = ExchangeDownloader().download("ETH/USDT:USDT", "15m", 2500)
df = pd.read_pickle(user_data / "data" / "ETH:USDT-15m.pkl")[-8000:]
dataset = TradeDataset(df, task.seq_len, task.pred_len)

get_dafe(model, dataset, 4000, 512)
exit()

for i in range(100):
    index = len(dataset) - int(input("s: "))
    pred, y = predict(model, dataset, index, device=device)

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(y=pred[:, 0], name="pred"), row=1, col=1)
    fig.add_trace(go.Scatter(y=y[:, 0], name="y"), row=1, col=1)
    fig.show()


exit()

while 1:
    index = len(df) - pred_len - int(input("s: "))
    df["pred"] = np.nan
    pred = predict(model, df, index, seq_len, pred_len, device=device)
    df.loc[pred.index, "pred"] = pred["close"]
    show(df[index - seq_len : index + pred_len])
exit()

# while 1:
#     start = len(df) - pred_len - int(input("s: "))
#     end = len(df) - pred_len - int(input("e: "))
#
#     for index in range(start, end):
#         df["pred"] = np.nan
#         pred = predict(model, df, index, seq_len, pred_len, device=device)
#         df.loc[pred.index, "pred"] = pred["close"]
#         show(df[index - seq_len : index + pred_len])
# exit()


# df.to_pickle("eth-signal.pkl")

show(df[start:end])

exit()
ExchangeDownloader
for i in [6, 12, 16, 20, 24]:  # [6, 12, 16, 20, 24]
    signal = genf(model, df, start, end, seq_len, pred_len, window_size=i)
    df.iloc[start : end + 1, df.columns.get_loc("signal")] = signal
    show(df[start:end])

exit()

# for i in range(start, end - pred_len, window + 1):
#     pred = predict(model, df, i, seq_len, pred_len, window)
#     df.iloc[i : i + window, df.columns.get_loc("pred")] = pred["close"]
#     print(i)

show(df[start:end])
exit()

# df = ExchangeDownloader().download("ETH/USDT:USDT", timerange, 5000)
#
# df["signal"] = 0.0
# start = seq_len
# end = len(df) - 1
# signal = genf(model, df, start, end, seq_len, pred_len)
# df.iloc[start : end + 1, df.columns.get_loc("signal")] = signal
#
#
# # df = pd.read_pickle("eth-signal.pkl")
# df["action"] = 0
# df["profit"] = 0
#
# pred = len(df) - seq_len
# signals = []
# autogena(df, 0, len(df) - pred, 64)
#
# profit = 0
# position = None
# quantity = 1
#
# for i in range(len(df) - pred, len(df) - 1):
#     current = df.index[i]
#     signal = gena(rmodel, df, i, seq_len, device)
#     df.loc[current, "action"] = signal
#
#     if signal == 1 and position is None:
#         position = df.loc[current, "close"]
#         signals.append({"time": current, "s": signal, "op": "buy"})
#
#     elif signal == 2 and position is not None:
#         sell_price = df.loc[current, "close"]
#         trade_profit = (sell_price - position) * quantity
#         profit += trade_profit
#         position = None
#
#         signals.append({"time": current, "s": signal, "op": "sell"})
#
#     df.loc[current, "profit"] = profit
#
#     print(current, profit)
#
# df = df[-pred:]
#
# show(df, signals)
# exit()

df = pd.read_pickle("eth-signal.pkl")
df["action"] = 0
seq_len = 64


autogena(df, 0, len(df), 32)

print()

print(
    (df["action"] == 0).sum(),
    "/",
    (df["action"] == 1).sum(),
    "/",
    (df["action"] == 2).sum(),
)

df.to_pickle("eth-signal-bhs.pkl")

exit()

# while 1:
#     df["pred"] = np.nan
#     index = len(df) - seq_len - pred_len - int(input(": "))
#     pred, ls = predict(model, df, index, seq_len, label_len, pred_len, device)
#     df.loc[pred.index, "pred"] = pred
#     show(df)
#     print(ls.item())
# exit()

signals = []

cp = False
ms = False
mx = 0
index = 0
for i in range(len(df) - seq_len):
    signal = genf(model, df, index, seq_len, pred_len, device)
    current = df.index[index + seq_len - 1]
    df.loc[current, "signal"] = signal

    print(df[current])
    exit()

    if signal > mx:
        mx = signal
        cp = True

    if signal < mx and mx > 3.5 and cp:
        signals.append({"time": current, "s": signal, "op": "buy"})
        cp = False
        ms = True

    if signal < -1 and mx != 0 and ms:
        signals.append({"time": current, "s": signal, "op": "sell"})
        mx = 0
        ms = False

    index += 1

    print(i)

show(df[seq_len:], signals)
exit()


index = len(df) - seq_len - pred_len
for i in range(20):
    pred = predict(model, df, index, seq_len, pred_len).iloc[:20]
    df.loc[pred.index, "pred"] = pred
    index -= 22

show(df)
exit()

# index = 1
# for i in range(50):
#     df = genf(model, df, index, seq_len, label_len, pred_len)
#     index += 1
#     print(i)

for i in range(25):
    index = 256 + 5 * i
    dfa = predict(model, df, index, seq_len, pred_len)
    show(dfa)

exit()

index = 50
for i in range(4200):
    df = predict(model, df, index, seq_len, pred_len)
    index += 2
    print(i)

del df["target"]
del df["pred"]

show(df)
df = ExchangeDownloader().download("ETH/USDT:USDT", timerange, 1000)
