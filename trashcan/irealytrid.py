import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ta
from pandas import DataFrame
from plotly.subplots import make_subplots


def get_targets(df: DataFrame) -> DataFrame:
    df["ma"] = ta.trend.sma_indicator(df["close"], window=10)
    df["roc"] = ta.momentum.roc(df["close"], window=4)
    df["macd"] = ta.trend.macd(df["close"], window_slow=26, window_fast=12)
    df["macdsignal"] = ta.trend.macd_signal(df["close"], window_slow=26, window_fast=12)
    df["macdhist"] = ta.trend.macd_diff(df["close"], window_slow=26, window_fast=12)

    df["momentum"] = df["close"] - df["close"].shift(4)
    df["rsi"] = ta.momentum.rsi(df["close"], window=10)

    df["bb_upperband"] = ta.volatility.bollinger_hband(df["close"], window=20)
    df["bb_middleband"] = ta.volatility.bollinger_mavg(df["close"], window=20)
    df["bb_lowerband"] = ta.volatility.bollinger_lband(df["close"], window=20)

    df["cci"] = ta.trend.cci(df["high"], df["low"], df["close"], window=20)
    df["stoch"] = ta.momentum.stoch(df["high"], df["low"], df["close"], window=14)
    df["atr"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=14
    )
    df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
    df["ma_100"] = ta.trend.sma_indicator(df["close"], window=100)

    def normalize(s: pd.Series, window: int) -> pd.Series:
        r = s.rolling(window)
        return (s - r.mean()) / (r.std() + 1e-8)

    df["normalized_stoch"] = normalize(df["stoch"], 14)
    df["normalized_atr"] = normalize(df["atr"], 14)
    df["normalized_obv"] = normalize(df["obv"], 14)
    df["normalized_ma"] = normalize(df["close"], 10)
    df["normalized_macd"] = normalize(df["macd"], 26)
    df["normalized_roc"] = normalize(df["roc"], 2)
    df["normalized_momentum"] = normalize(df["momentum"], 4)
    df["normalized_rsi"] = normalize(df["rsi"], 10)
    df["normalized_cci"] = normalize(df["cci"], 20)

    bb_width = df["bb_upperband"] - df["bb_lowerband"]
    df["normalized_bb_width"] = normalize(bb_width, 20)

    trend_strength = abs(df["ma"] - df["close"])
    rolling_mean = trend_strength.rolling(window=14).mean()
    rolling_stddev = trend_strength.rolling(window=14).std()
    strong_trend_threshold = rolling_mean + 1.5 * rolling_stddev

    w = [
        0.54347,
        0.82226,
        0.56675,
        0.77918,
        0.98488,
        0.31368,
        0.75916,
        0.09226,
        0.85667,
    ]

    df["w_momentum"] = w[3] * (1 + 0.5 * (trend_strength / strong_trend_threshold))
    df["w_momentum"] = df["w_momentum"].clip(lower=w[3], upper=w[3] * 2)

    df["S"] = (
        w[0] * df["normalized_ma"]
        + w[1] * df["normalized_macd"]
        + w[2] * df["normalized_roc"]
        + w[3] * df["normalized_rsi"]
        + w[4] * df["normalized_bb_width"]
        + w[5] * df["normalized_cci"]
        + df["w_momentum"] * df["normalized_momentum"]
        + w[8] * df["normalized_stoch"]
        + w[7] * df["normalized_atr"]
        + w[6] * df["normalized_obv"]
    )

    df["R"] = 0
    df.loc[df["close"] > df["bb_upperband"], "R"] = 1
    df.loc[df["close"] < df["bb_lowerband"], "R"] = -1
    buffer_pct = 0.01

    df["R2"] = np.where(
        df["close"] > df["ma_100"] * (1 + buffer_pct),
        1,
        np.where(df["close"] < df["ma_100"] * (1 - buffer_pct), -1, np.nan),
    )

    bb_width = (df["bb_upperband"] - df["bb_lowerband"]) / df["bb_middleband"]
    df["V_mean"] = 1 / (bb_width + 1e-8)
    df["V2_mean"] = 1 / (df["atr"] + 1e-8)

    mean_v = df["V_mean"].rolling(50).mean()
    std_v = df["V_mean"].rolling(50).std()
    df["V_norm"] = (df["V_mean"] - mean_v) / std_v
    df["V_norm"] = df["V_norm"].fillna(0)

    mean_v2 = df["V2_mean"].rolling(50).mean()
    std_v2 = df["V2_mean"].rolling(50).std()
    df["V2_norm"] = (df["V2_mean"] - mean_v2) / std_v2
    df["V2_norm"] = df["V2_norm"].fillna(0)

    upper_threshold = 1.0
    lower_threshold = -1.0

    df["V"] = np.where(
        df["V_norm"] > upper_threshold,
        1,
        np.where(df["V_norm"] < lower_threshold, -1, np.nan),
    )
    df["V2"] = np.where(
        df["V2_norm"] > upper_threshold,
        1,
        np.where(df["V2_norm"] < lower_threshold, -1, np.nan),
    )

    df["V"] = df["V"].ffill()
    df["V2"] = df["V2"].ffill()

    print(df["R"][100:].isna().sum())
    print(df["R2"][100:].isna().sum())
    print(df["V"][100:].isna().sum())
    print(df["V2"][100:].isna().sum())

    df["T"] = df["S"] * df["R"] * df["R2"] * df["V"] * df["V2"]

    df["&-target"] = df["T"].shift(-1)
    return df


df = pd.read_pickle("/home/mastru/pg/trading/user_data/logs/PORTAL:USDT:USDT.pkl")
# df = get_targets(df)

# 2) Создаем фигуру с двумя панелями (верх — цена, низ — индикатор T)
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.7, 0.3],
    vertical_spacing=0.05,
    subplot_titles=["Цена закрытия", "Индикатор T"],
)

# 3) График цены
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["close"],
        mode="lines",
        name="Close Price",
        line=dict(color="red", width=1.5),
    ),
    row=1,
    col=1,
)

# 4) Для T рисуем бары: зелёные, если T>0 (покупка), и красные, если T<0 (продажа)
colors = df["T"].apply(lambda x: "green" if x > 0 else "red")
fig.add_trace(
    go.Bar(x=df.index, y=df["T"], marker_color=colors, name="T Indicator"), row=2, col=1
)

# 5) Горизонтальная линия на уровне 0 во второй панели
fig.add_shape(
    type="line",
    x0=df.index.min(),
    x1=df.index.max(),
    y0=0,
    y1=0,
    line=dict(color="black", width=1, dash="dash"),
    row=2,
    col=1,
)

# 6) Общие настройки
fig.update_layout(
    height=600,
    showlegend=False,
    title="Tralalelo Tralala",
    xaxis=dict(rangeslider=dict(visible=False)),  # убираем range-slider, если не нужен
)

# 7) Подписи осей
fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
fig.update_yaxes(title_text="T value", row=2, col=1)

fig.show()
