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
            if not batch:
                break
            all_ohlcv = batch + all_ohlcv
            print(len(all_ohlcv))

        df = pd.DataFrame(
            all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/Moscow")
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep="first")]

        return df


if __name__ == "__main__":
    pair = "XRP/USDT:USDT"
    timeframe = "5m"
    df = ExchangeDownloader().download(pair, timeframe, 4_000_000)
    pairname = pair.split(":")[0].replace("/", ":")
    df.to_pickle(f"data/{pairname}-{timeframe}.pkl")
