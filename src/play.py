from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from tsfs.core import TaskConfig
from tsfs.models.duet import DUET, DUETConfig

from .data.dataset import TradeDataset
from .parseset import get_dafe, predict
from .utils import init_device, set_seeds, user_data_init

torch.autograd.set_detect_anomaly(False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device.upper())


user_data = Path("user_data")
runname = "sambucha"

set_seeds(2003)
init_device(device)
user_data_init(user_data)

task = TaskConfig(1024, 32, enc_in=4, c_out=3)

config = DUETConfig()
config.apply(task)

model = DUET(config).to(device)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📊 Params: {params:,}")


checkpoint = user_data / "state" / (runname + ".pth")
if checkpoint.is_file():
    cpdata = torch.load(checkpoint, weights_only=False)
    model.load_state_dict(cpdata["model"])
    print(f"✅ Checkpoint m loaded. Loss {cpdata['loss']:.6f}")

# df = ExchangeDownloader().download("ETH/USDT:USDT", "15m", 2500)
df = pd.read_pickle(user_data / "data" / "BTC:USDT-1m.pkl")[-24000:]
dataset = TradeDataset(df, task.seq_len, task.pred_len)

get_dafe(model, dataset, 20000, 512, device)
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
