import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data.dataset import TradeDataset
from .scaler import Robust


@torch.no_grad()
def predict(
    model,
    dataset: TradeDataset,
    batch_size,
    device="cuda",
):
    model.eval()
    preds = np.empty(len(dataset), dtype=np.float32)
    profits = np.empty(len(dataset), dtype=np.float32)

    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
    )
    df = dataset.df[dataset.seq_len :]

    i = 0

    aqua = []
    batch_bar = tqdm(loader, desc="Predict", unit="batch")
    for x, y, mark, _ in batch_bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mark = mark.to(device, non_blocking=True)
        with autocast(device):
            outputs, _ = model(x, mark)
            aqua.append(torch.nn.functional.mse_loss(outputs, y).item())
        output = y[:, 0, 0].cpu().numpy()
        output = outputs[:, 0, 0].cpu().numpy()
        step = len(output)
        preds[i : i + step] = output
        i += step

    vl = sum(aqua) / len(aqua)
    print(vl)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.6, 0.4],
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Исторические данные",
        ),
        row=1,
        col=1,
    )

    changes = np.where(np.diff(preds >= 0.5))[0] + 1
    segments = np.split(range(len(df)), changes)

    res = 0
    for segment in segments:
        price0 = df.iloc[segment[0], 3]
        prices = df.iloc[segment, 3]
        signal = preds[segment[0]]

        lir = 0
        for i, price in enumerate(prices):
            lir = (price0 / price - 1) * (-1 if signal >= 0.5 else 1) * 100

        res += lir

        profits[segment] = res

        fig.add_trace(
            go.Scatter(
                x=df.index[segment],
                y=df.iloc[segment, 3],
                mode="lines",
                line=dict(color="green" if signal >= 0.5 else "red"),
                showlegend=False,
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
    print(res)

    # fig.add_trace(
    #     go.Scatter(x=df.index, y=preds),
    #     row=2,
    #     col=1,
    # )

    fig.add_trace(
        go.Scatter(x=df.index, y=profits),
        row=2,
        col=1,
    )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        template="plotly_dark",
    )
    fig.show()


@torch.no_grad()
def get_dafe(
    model,
    dataset: TradeDataset,
    val_size: int,
    batch_size: int,
    device="cuda",
):
    rb = Robust()
    preds = []
    targets = []

    model.eval()

    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
    )
    df = dataset.df[dataset.seq_len :]

    batch_bar = tqdm(loader, desc="Dafe Loss", unit="batch")
    for x, y, mark, context in batch_bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mark = mark.to(device, non_blocking=True)
        with autocast(device):
            outputs, _ = model(x, mark)
            loss = (outputs - y).abs().mean(dim=(-1, -2))
        for pred, target, cont in zip(outputs, y, context[:, 0]):
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
            cont = cont.cpu().numpy()

            pred = rb.unscale(pred, cont)
            target = rb.unscale(target, cont)

            preds.append(pred[31, 0])
            targets.append(target[31, 0])

    preds = np.hstack(preds)
    targets = np.hstack(targets)

    fig = make_subplots(
        rows=1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[1.0],
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Исторические данные",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=targets, mode="lines", line=dict(color="#47ff57", width=2)
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=preds, mode="lines", line=dict(color="#ff4757", width=2)
        ),
        row=1,
        col=1,
    )
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        template="plotly_dark",
    )
    fig.show()


def genf(
    model,
    df,
    start_index,
    end_index,
    seq_len,
    pred_len,
    batch_size=32,
    window_size=12,
    device="cuda",
):
    data = df[["close", "high", "low", "volume"]].values
    fdidx = df.index
    additional = pd.date_range(
        start=fdidx[-1] + pd.Timedelta(fdidx.freq),
        periods=pred_len,
        freq=fdidx.freq,
        tz=fdidx.tz,
    )
    fdidx = fdidx.append(additional)
    feats = TradeDataset._gen_time_features(fdidx)

    num_samples = end_index - start_index + 1
    batch_inputs_x = torch.empty((num_samples, seq_len, 4), device=device)
    batch_inputs_mark = torch.empty((num_samples, seq_len + pred_len, 4), device=device)
    min_max_values = []

    for i, index in enumerate(range(start_index, end_index + 1)):
        s_begin = index - seq_len
        s_end = index
        r_end = s_end + pred_len

        if s_begin < 0 or s_end > len(data):
            raise IndexError(f"{index} [- {seq_len}]")

        input_seq = data[s_begin:s_end]

        price, pmn, pmx = scale_robust(input_seq[:, 0:3])
        volume, _, _ = scale_robust(input_seq[:, 3:4])

        min_max_values.append((pmn, pmx))

        x = np.concat([price, volume], axis=1)
        mark = feats[s_begin:r_end]

        batch_inputs_x[i, :, :] = torch.from_numpy(x).to(device, non_blocking=True)
        batch_inputs_mark[i, :, :] = torch.from_numpy(mark).to(
            device, non_blocking=True
        )

        if i % 100 == 0:
            print(i)

    predictions = torch.empty((num_samples, pred_len, 4), device=device)
    model.eval()
    with torch.no_grad(), autocast(device):
        for i in range(0, batch_inputs_x.size(0), batch_size):
            batch_x = batch_inputs_x[i : i + batch_size]
            batch_mark = batch_inputs_mark[i : i + batch_size]
            outputs, _ = model(batch_x, batch_mark)
            predictions[i : i + outputs.size(0)] = outputs

            if i % 100 == 0:
                print(i)

    index = 0
    deltas = torch.empty(num_samples, device=device)
    for pred, (pmn, pmx), x in zip(predictions, min_max_values, batch_inputs_x):
        pred = unscale_robust(pred[:, 0:3], pmn, pmx)

        close_delta = pred[:window_size, 0] / pred[0, 0] - 1
        deltas[index] = close_delta.mean()

        index += 1

        if index % 100 == 0:
            print(index)

    alpha = 0.4
    raw_signal = deltas.cpu().numpy() * 10
    raw_signal = np.sign(raw_signal) * (np.abs(raw_signal) ** alpha)
    return raw_signal


def gena(
    model,
    dataset: TradeDataset,
    val_size: int,
    batch_size: int,
    device="cuda",
):
    model.eval()
    losses = []

    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
    )
    df = dataset.df[dataset.seq_len :]

    batch_bar = tqdm(loader, desc="Dafe Loss", unit="batch")
    for x, y, mark, context in batch_bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mark = mark.to(device, non_blocking=True)
        with autocast(device):
            outputs, _ = model(x, mark)
            loss = (outputs - y).abs().mean(dim=(-1, -2))
        loss = torch.tensor(0.0, device=device)
        for pred, target, cont in zip(outputs, y, context[:, 0]):
            losses.append(abs(pred - target).mean().cpu().numpy())

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.5, 0.25, 0.25],
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Исторические данные",
        ),
        row=1,
        col=1,
    )
    split_point = len(losses) - val_size
    train_losses = losses[:split_point]
    val_losses = losses[split_point:]

    fig.add_trace(
        go.Scatter(
            x=df.index[:split_point],
            y=train_losses,
            mode="lines",
            line=dict(color="#ff4757", width=2),
            name="Train Loss",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index[split_point:],
            y=val_losses,
            mode="lines",
            line=dict(color="#2ed573", width=2),
            name="Validation Loss",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=train_losses,
            name="Train Loss Distribution",
            nbinsx=400,
            marker_color="red",
            histnorm="probability",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=val_losses,
            name="Val Loss Distribution",
            nbinsx=400,
            marker_color="green",
            histnorm="probability",
        ),
        row=3,
        col=1,
    )
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        template="plotly_dark",
    )
    fig.show()


def autogena(df, start, end, seq_len):
    state = "min"
    index = start

    timestamp = df.index[0]

    while index + seq_len <= end:
        frame = df.iloc[index : index + seq_len]["close"]
        if state == "min":
            timestamp = frame.idxmin()
            df.loc[timestamp, "action"] = 1
            state = "max"
        else:
            timestamp = frame.idxmax()
            df.loc[timestamp, "action"] = 2
            state = "min"

        index = df.index.get_loc(timestamp) + 1


# def autogena(df, start, end, seq_len):
#     state = "min"
#     index = start
#
#     timestamp = df.index[0]
#
#     while index + seq_len <= end:
#         frame = df.iloc[index : index + seq_len]["close"]
#         if state == "min":
#             lstamp = timestamp
#             timestamp = frame.idxmin()
#             df.loc[lstamp:timestamp, "action"] = 0
#             state = "max"
#         else:
#             lstamp = timestamp
#             timestamp = frame.idxmax()
#             signal = df.loc[timestamp, "close"] / df.loc[lstamp, "close"] - 1
#             if signal > 0.01:
#                 df.loc[lstamp:timestamp, "action"] = signal
#             state = "min"
#
#         index = df.index.get_loc(timestamp) + 1
