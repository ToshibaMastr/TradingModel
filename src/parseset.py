import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
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


def predict(model, df, index, seq_len, pred_len):
    s_begin = index
    s_end = s_begin + seq_len
    r_end = s_end + pred_len

    if s_end > len(df):
        raise IndexError("DataFrame")

    data = df[["close", "high", "low", "volume"]]
    input_seq = data[s_begin:s_end]
    target_seq = data[s_end:r_end]

    price, pmn, pmx = scale(input_seq[["close", "high", "low"]])
    volume, vmn, vmx = scale(input_seq[["volume"]])

    tprice, _, _ = scale(target_seq[["close", "high", "low"]], pmn, pmx)
    tvolume, _, _ = scale(target_seq[["volume"]], vmn, vmx)

    x = pd.concat([price, volume], axis=1).values
    y = pd.concat([tprice, tvolume], axis=1).values

    x = torch.FloatTensor(x).unsqueeze(0).to("cuda")
    y = torch.FloatTensor(y).unsqueeze(0).to("cuda")

    model.eval()
    with torch.no_grad(), autocast("cuda"):
        outputs, _ = model(x)
        loss = F.mse_loss(outputs[:, :, 0], y[:, :, 0], reduction="mean")

        outputs = outputs.squeeze(0).cpu().numpy()

    pred_close = unscale(outputs[:, 1], pmn, pmx)
    pred_index = df.index[s_end:r_end]
    return pd.Series(pred_close, index=pred_index, name="pred"), loss

    # columns = ["close", "high", "low", "volume"]
    # output_df = pd.DataFrame(outputs, columns=columns)

    # output_df[["close", "high", "low"]] = unscale(
    #     output_df[["close", "high", "low"]], pmn, pmx
    # )
    # output_df[["volume"]] = unscale(output_df[["volume"]], vmn, vmx)
    # output_df.index = df.index[s_end:r_end]


def genf(model, df, index, seq_len, pred_len):
    s_begin = index
    s_end = s_begin + seq_len

    if s_end > len(df):
        raise IndexError("DataFrame")

    data = df[["close", "high", "low", "volume"]]
    input_seq = data[s_begin:s_end]

    price, pmn, pmx = scale(input_seq[["close", "high", "low"]])
    volume, vmn, vmx = scale(input_seq[["volume"]])

    x = pd.concat([price, volume], axis=1).values
    x = torch.FloatTensor(x).unsqueeze(0).to("cuda")

    model.eval()
    with torch.no_grad(), autocast("cuda"):
        outputs, _ = model(x)
        outputs = outputs.squeeze(0).cpu().numpy()[:20, 0]

    return sum(outputs - outputs[0])
