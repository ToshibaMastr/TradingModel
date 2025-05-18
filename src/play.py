from pathlib import Path

import torch

from autoformer.model import Autoformer

from .download import ExchangeDownloader
from .parseset import genf, predict, show

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using ", device)

seq_len = 192
label_len = 48
pred_len = 24
timerange = "5m"

checkpoint = Path("state") / f"S{seq_len}P{pred_len}:{timerange}.pth"


model = Autoformer(4, 4, 1)
model.to(device)


if checkpoint.is_file():
    cpdata = torch.load(checkpoint)
    model.load_state_dict(cpdata["model"])
    best_loss = cpdata["loss"]
    print(f"✅ Checkpoint loaded. Loss {best_loss}")


df = ExchangeDownloader().download("ETH/USDT:USDT", timerange, 1000)
df["target"] = df["close"].rolling(window=5, min_periods=1).mean()

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
for i in range(len(df) - seq_len - 100):
    signal = genf(model, df, index, seq_len, label_len, pred_len, device)
    current = df.index[index + seq_len - 1]
    df.loc[current, "signal"] = signal

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
