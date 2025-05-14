from pathlib import Path

import torch

from .download import ExchangeDownloader
from .duet.config import DUETConfig
from .duet.model import DUETModel
from .parseset import genf, predict, show

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using ", device)

seq_len = 256
pred_len = 48
timerange = "5m"

checkpoint = Path("state") / f"S{seq_len}P{pred_len}:{timerange}.pth"


config = DUETConfig()
config.seq_len = seq_len
config.pred_len = pred_len
config.enc_in = 4
config.CI = 1
model = DUETModel(config)
model.to(device)


if checkpoint.is_file():
    cpdata = torch.load(checkpoint)
    model.load_state_dict(cpdata["model"])
    best_loss = cpdata["loss"]
    print(f"✅ Checkpoint loaded. Loss {best_loss}")


df = ExchangeDownloader().download("ADA/USDT:USDT", timerange, 1000)


# while 1:
#     df["pred"] = np.nan
#
#     index = len(df) - seq_len - pred_len - int(input("index: "))
#     pred, ls = predict(model, df, index, seq_len, pred_len, device)
#     df.loc[pred.index, "pred"] = pred
#
#     show(df)
#     print(ls)
#
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
