import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots
from torch import autocast

from .dataset import TradeDataset
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


def predict(model, data, index, seq_len, label_len, pred_len, device):
    s_begin = index
    s_end = s_begin + seq_len
    r_begin = s_end - label_len
    r_end = s_end + pred_len

    if r_end > len(data):
        raise IndexError("DataFrame")

    time_feats = TradeDataset._gen_time_features(data.index)

    input_seq = data[s_begin:s_end]
    target_seq = data[r_begin:r_end]

    price, pmn, pmx = scale(input_seq[["close", "high", "low"]])
    volume, vmn, vmx = scale(input_seq[["volume"]])

    tprice, _, _ = scale(target_seq[["close", "high", "low"]], pmn, pmx)
    tvolume, _, _ = scale(target_seq[["volume"]], vmn, vmx)

    x = pd.concat([price, volume], axis=1).values
    y = pd.concat([tprice, tvolume], axis=1).values

    x_mark = time_feats[s_begin:s_end]
    y_mark = time_feats[r_begin:r_end]

    x = torch.FloatTensor(x).unsqueeze(0).to(device)
    y = torch.FloatTensor(y).unsqueeze(0).to(device)

    x_mark = torch.FloatTensor(x_mark).unsqueeze(0).to(device)
    y_mark = torch.FloatTensor(y_mark).unsqueeze(0).to(device)

    dec_inp = torch.zeros_like(y[:, -pred_len:, :]).float()
    dec_inp = torch.cat([y[:, :label_len, :], dec_inp], dim=1).float().to(y.device)

    model.eval()
    with torch.no_grad(), autocast(device):
        outputs, _ = model(x, x_mark, dec_inp, y_mark)

        outputs = outputs[:, -pred_len:, :]
        y = y[:, -pred_len:, :].to(outputs.device)

        loss = F.mse_loss(outputs, y, reduction="mean")

        outputs = outputs[0, :, 0].cpu().numpy()

    pred_close = unscale(outputs, pmn, pmx)
    pred_index = df.index[s_end:r_end]
    return pd.Series(pred_close, index=pred_index, name="pred"), loss


def genf(model, data, index, seq_len, label_len, pred_len, device):
    s_begin = index
    s_end = s_begin + seq_len
    r_begin = s_end - label_len
    r_end = s_end + pred_len

    if r_end > len(data):
        raise IndexError("DataFrame")

    time_feats = TradeDataset._gen_time_features(data.index)

    input_seq = data[s_begin:s_end]
    target_seq = data[r_begin:r_end]

    price, pmn, pmx = scale(input_seq[["close", "high", "low"]])
    volume, vmn, vmx = scale(input_seq[["volume"]])

    tprice, _, _ = scale(target_seq[["close", "high", "low"]], pmn, pmx)
    tvolume, _, _ = scale(target_seq[["volume"]], vmn, vmx)

    x = pd.concat([price, volume], axis=1).values
    y = pd.concat([tprice, tvolume], axis=1).values

    x_mark = time_feats[s_begin:s_end]
    y_mark = time_feats[r_begin:r_end]

    x = torch.FloatTensor(x).unsqueeze(0).to(device)
    y = torch.FloatTensor(y).unsqueeze(0).to(device)

    x_mark = torch.FloatTensor(x_mark).unsqueeze(0).to(device)
    y_mark = torch.FloatTensor(y_mark).unsqueeze(0).to(device)

    dec_inp = torch.zeros_like(y[:, -pred_len:, :]).float()
    dec_inp = torch.cat([y[:, :label_len, :], dec_inp], dim=1).float().to(y.device)

    model.eval()
    with torch.no_grad(), autocast(device):
        outputs, _ = model(x, x_mark, dec_inp, y_mark)

        outputs = outputs[:, -pred_len:, :]
        y = y[:, -pred_len:, :].to(outputs.device)

        outputs = outputs[0, :, 0].cpu().numpy()

    return sum(outputs - outputs[0])
