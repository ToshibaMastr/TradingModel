import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data.dataset import TradeDataset


def show(data, signals: list[dict] = []):
    data.loc[:, "open"] = data["close"].shift(1)

    has_signal = "signal" in data.columns
    has_pred = "pred" in data.columns

    colors = np.where(
        data["close"] > data["open"], "rgba(0,255,0,0.5)", "rgba(255,0,0,0.5)"
    )

    row_heights = [0.7, 0.3] if has_signal else [1.0]
    rows = 2 if has_signal else 1
    specs = [[{"secondary_y": True}]] + ([[{}]] if has_signal else [])

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
        specs=specs,
    )

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="Исторические данные",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data["volume"],
            name="Объем",
            marker=dict(color=colors),
            opacity=0.4,
            yaxis="y2",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    for signal in signals:
        is_buy = signal["op"] == "buy"
        fig.add_annotation(
            x=signal["time"],
            y=data.loc[signal["time"], "high"],
            text="▲" if is_buy else "▼",
            showarrow=True,
            arrowhead=1,
            arrowcolor="green" if is_buy else "red",
            font=dict(color="green" if is_buy else "red", size=14),
            row=1,
            col=1,
        )

    if has_signal:
        fig.add_trace(
            go.Bar(x=data.index, y=data["signal"], name="Сигнал"),
            row=2,
            col=1,
        )

    if has_pred:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["pred"],
                mode="lines",
                line=dict(color="red", width=4),
                name="Прогноз",
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        yaxis_title="Цена",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        template="plotly_dark",
        yaxis=dict(
            title="Цена",
            anchor="x",
        ),
        yaxis2=dict(
            title="Объем",
            overlaying="y",
            anchor="x",
            side="right",
            showgrid=False,
        ),
    )

    fig.show()


@torch.no_grad()
def get_dafe(
    model,
    dataset: TradeDataset,
    val_size,
    batch_size,
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
    for x, y, mark, _ in batch_bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mark = mark.to(device, non_blocking=True)
        with autocast(device):
            outputs, _ = model(x, mark)
            loss = (outputs - y).abs().mean(dim=(-1, -2))
        losses += loss.tolist()

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


def predict(
    model,
    dataset: TradeDataset,
    index: int,
    window_size=None,
    device="cuda",
):
    if window_size is None:
        window_size = dataset.seq_len
    elif window_size > dataset.pred_len:
        raise IndexError("window")

    x, y, mark, _ = dataset[index]

    x = x.to(device, non_blocking=True).unsqueeze(0)
    y = y.to(device, non_blocking=True).unsqueeze(0)
    mark = mark.to(device, non_blocking=True).unsqueeze(0)

    model.eval()
    with torch.no_grad(), autocast(device):
        outputs, _ = model(x, mark)
        outputs = outputs.cpu().numpy()[0]

    return outputs, y.cpu().numpy()[0]


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


def gena(model, data, index, seq_len, device):
    s_begin = index - seq_len
    s_end = index

    if s_end > len(data):
        raise IndexError("DataFrame")

    data = data[["close", "signal", "volume", "action"]]
    input_seq = data[s_begin:s_end]

    price, _, _ = scale_robust(input_seq[["close"]])
    volume, _, _ = scale_robust(input_seq[["volume"]])
    signal = input_seq[["signal"]]

    frame = pd.concat([price, signal, volume], axis=1).values
    x = input_seq["action"].values

    x = torch.LongTensor(x).unsqueeze(0).to(device)
    frame = torch.FloatTensor(frame).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad(), autocast(device):
        logits, _ = model(x, frame)

    return torch.argmax(logits, dim=-1).cpu().numpy()[0][-1]


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
