from pathlib import Path

import ccxt
import pandas as pd


class ExchangeDownloader:
    def __init__(self):
        self.exchange = ccxt.binance()

    def download(self, pair: str, timeframe: str, length: int):
        tf_seconds = self.exchange.parse_timeframe(timeframe)
        tf_ms = tf_seconds * 1000 * 1000

        all_ohlcv = self.exchange.fetch_ohlcv(pair, timeframe, limit=1000)
        while len(all_ohlcv) < length:
            oldest_ts = all_ohlcv[0][0]
            batch = self.exchange.fetch_ohlcv(
                pair, timeframe, since=oldest_ts - tf_ms, limit=1000
            )
            if not batch or oldest_ts == batch[0][0]:
                break
            all_ohlcv = batch + all_ohlcv
            print(len(all_ohlcv), all_ohlcv[0][0], pair)

        df = pd.DataFrame(
            all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/Moscow")
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep="first")].asfreq(self.tf2freq(timeframe))

        return df

    @staticmethod
    def tf2freq(timeframe: str) -> str:
        if timeframe.endswith("m"):
            return timeframe[:-1] + "min"
        elif timeframe.endswith("h"):
            return timeframe[:-1] + "H"
        elif timeframe.endswith("d"):
            return timeframe[:-1] + "D"
        return timeframe


def download(pair: str, timeframe: str, length: int = 4_000_000):
    pair += "/USDT:USDT"

    pairname = pair.split(":")[0].replace("/", ":")
    file = Path("data") / f"{pairname}-{timeframe}.pkl"

    if not file.exists():
        df = ExchangeDownloader().download(pair, timeframe, length)
        df.to_pickle(file)


if __name__ == "__main__":
    timeframe = "15m"

    for symbol in [
        "BTC",
        "ETH",
        "XRP",
        "BNB",
        "SOL",
        "USDC",
        "DOGE",
        "TRX",
        "ADA",
        "HYPE",
        "SUI",
        "LINK",
        "AVAX",
        "XLM",
        "BCH",
        "TON",
        "HBAR",
        "LTC",
        "DOT",
        "XMR",
    ]:
        download(symbol, timeframe)
