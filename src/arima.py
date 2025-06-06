import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA

from .download import ExchangeDownloader

df = ExchangeDownloader().download("ETH/USDT:USDT", "15m", 1000)
np.random.seed(42)
n = len(df)
time = np.arange(n)
trend = 0.1 * time
noise = np.random.normal(loc=0, scale=5, size=n)
series = trend + noise
df["close"] = series

model = ARIMA(df["close"], order=(5, 0, 2))
model_fit = model.fit()

print(model_fit.summary())

forecast = model_fit.forecast(steps=128)

fdidx = df.index
additional = pd.date_range(
    start=fdidx[-1] + pd.Timedelta(fdidx.freq),
    periods=128,
    freq=fdidx.freq,
    tz=fdidx.tz
)
fdidx = fdidx.append(additional)

fig = go.Figure()
# fig.add_trace(go.Candlestick(
#     x=df.index,
#     open=df["open"],
#     high=df["high"],
#     low=df["low"],
#     close=df["close"]
# ))
fig.add_trace(go.Scatter(
    x=df.index,
    y=df["close"],
    mode='lines'
))
fig.add_trace(go.Scatter(
    x=additional,
    y=forecast,
    mode='lines+markers',
    name='Прогноз',
    line=dict(color='red', dash='dash')
))
fig.update_layout(
    title='ARIMA Forecast',
    xaxis_title='Time',
    yaxis_title='Value',
    template='plotly_white'
)
fig.show()
