from pathlib import Path

import torch

from duet.config import DUETConfig
from duet.model import DUETModel
from trader.model import TradeModel

from .download import ExchangeDownloader
from .parseset import autogena, gena, genf, show

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using ", device)

if device == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


seq_len = 256
pred_len = 24
timerange = "5m"


config = DUETConfig()
config.seq_len = seq_len
config.pred_len = pred_len
config.enc_in = 4
model = DUETModel(config).to(device)

rmodel = TradeModel(3, 3, 64).to(device)

checkpoint = Path("state") / f"S{seq_len}P{pred_len}:{timerange}.pth"
if checkpoint.is_file():
    cpdata = torch.load(checkpoint)
    model.load_state_dict(cpdata["model"])
    print(f"✅ Checkpoint m loaded. Loss {cpdata['loss']:.6f}")

checkpoint_path = Path("state") / "TM_S32.pth"
if checkpoint_path.is_file():
    cpdata = torch.load(checkpoint_path, map_location=device)
    rmodel.load_state_dict(cpdata["model"])
    print(f"✅ Checkpoint r loaded. Loss {cpdata['loss']:.6f}")

df = ExchangeDownloader().download("ETH/USDT:USDT", timerange, 1000)

df["signal"] = 0.0
start = seq_len
end = len(df) - 1
signal = genf(model, df, start, end, seq_len, pred_len)
df.iloc[start : end + 1, df.columns.get_loc("signal")] = signal

df["action"] = 0
df["profit"] = 0

pred = len(df) - seq_len
signals = []
autogena(df, 0, len(df) - pred, 64)

profit = 0
position = None
quantity = 1

for i in range(len(df) - pred, len(df) - 1):
    current = df.index[i]
    signal = gena(rmodel, df, i, seq_len, device)
    df.loc[current, "action"] = signal

    if signal == 1 and position is None:
        position = df.loc[current, "close"]
        signals.append({"time": current, "s": signal, "op": "buy"})

    elif signal == 2 and position is not None:
        sell_price = df.loc[current, "close"]
        trade_profit = (sell_price - position) * quantity
        profit += trade_profit
        position = None

        signals.append({"time": current, "s": signal, "op": "sell"})

    df.loc[current, "profit"] = profit

    print(current, profit)

df = df[-pred:]

show(df, signals)
