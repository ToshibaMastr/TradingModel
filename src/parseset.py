import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch import autocast

from .utils import scale, unscale


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


def predict(model, data, index, seq_len, pred_len, window_size=None, device="cuda"):
    if window_size is None:
        window_size = pred_len
    elif window_size > pred_len:
        raise IndexError("window")

    s_begin = index - seq_len
    s_end = index
    r_end = s_end + window_size

    if s_end > len(data):
        raise IndexError("DataFrame")

    data = data[["close", "high", "low", "volume"]]
    input_seq = data[s_begin:s_end]

    price, pmn, pmx = scale(input_seq[["close", "high", "low"]])
    volume, vmn, vmx = scale(input_seq[["volume"]])

    x = pd.concat([price, volume], axis=1).values
    x = torch.FloatTensor(x).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad(), autocast(device):
        outputs, _ = model(x)
        outputs = outputs.squeeze(0).cpu().numpy()

    pred_index = data.index[s_end:r_end]
    price = unscale(outputs[:window_size, 0:3], pmn, pmx)
    volume = unscale(outputs[:window_size, 3], vmn, vmx).reshape(-1, 1)

    result = pd.DataFrame(
        data=np.hstack([price, volume]),
        index=pred_index,
        columns=["close", "high", "low", "volume"],
    )

    return result


def genf(
    model,
    data,
    start_index,
    end_index,
    seq_len,
    pred_len,
    batch_size=32,
    window_size=12,
    device="cuda",
):
    data = data[["close", "high", "low", "volume"]]

    num_samples = end_index - start_index + 1
    batch_inputs = torch.empty((num_samples, seq_len, 4), device=device)
    min_max_values = []

    for i, index in enumerate(range(start_index, end_index + 1)):
        s_begin = index - seq_len
        s_end = index

        if s_begin < 0 or s_end > len(data):
            raise IndexError(f"{index} [- {seq_len}]")

        input_seq = data.iloc[s_begin:s_end]

        price, pmn, pmx = scale(input_seq[["close", "high", "low"]])
        volume, _, _ = scale(input_seq[["volume"]])

        min_max_values.append((pmn, pmx))

        x = pd.concat([price, volume], axis=1).values
        batch_inputs[i, :, :] = torch.from_numpy(x).to(device, non_blocking=True)

        if i % 100 == 0:
            print(i)

    predictions = torch.empty((num_samples, pred_len, 4), device=device)
    model.eval()
    with torch.no_grad(), autocast(device):
        for i in range(0, batch_inputs.size(0), batch_size):
            batch_x = batch_inputs[i : i + batch_size]
            outputs, _ = model(batch_x)
            predictions[i : i + outputs.size(0)] = outputs

            if i % 100 == 0:
                print(i)

    index = 0
    deltas = torch.empty(num_samples, device=device)
    for pred, (pmn, pmx) in zip(predictions, min_max_values):
        pred_close = unscale(pred[-window_size:, 0], pmn, pmx)
        pred_high = unscale(pred[-window_size:, 1], pmn, pmx)
        pred_low = unscale(pred[-window_size:, 2], pmn, pmx)

        close_delta = pred_close / pred_close[0] - 1

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

    price, _, _ = scale(input_seq[["close"]])
    volume, _, _ = scale(input_seq[["volume"]])
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
