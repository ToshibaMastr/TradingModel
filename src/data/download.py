from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import cycle
from typing import Any

import httpx
import numpy as np
import pandas as pd

# Надо бы вынести в отдельный пакет...


class ExchangeDownloader:
    def __init__(self, workers: int = 5, timeout: float = 10.0):
        urls = cycle([f"https://api{i}.binance.com" for i in range(1, 5)])
        clients = [
            httpx.Client(base_url=next(urls), timeout=timeout) for i in range(workers)
        ]
        self.conns = cycle(clients)
        self.workers = workers

    def request(self, endpoint: str, params: dict, info: Any = None):
        client = next(self.conns)
        try:
            response = client.get(endpoint, params=params)
            response.raise_for_status()
            return (response.json(), info)
        except Exception as e:
            print(e)
            print(params)
            raise e

    def download(self, symbol: str, interval: str, length: int, limit: int = 1000):
        symbol = symbol.replace(":", "")

        step = self.interval2ms(interval)
        end = int(datetime.now().timestamp() * 1000)
        data, _ = self.request(
            "/api/v3/klines",
            {"symbol": symbol, "interval": interval, "startTime": 0, "limit": 1},
        )

        print(end, data[0][0])
        exit()

        length = min((end - data[0][0]) // step, length)

        ohlcv_array = np.empty((length, 6), dtype=np.float64)

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            collected = 0
            start = end - step * length

            for current in range(start, end, step * limit):
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": current,
                    "endTime": current + step * limit,
                    "limit": limit,
                }
                info = {"start": collected, "end": collected + limit}
                collected += limit

                future = executor.submit(self.request, "/api/v3/klines", params, info)
                futures.append(future)

            total = 0
            for future in as_completed(futures):
                data, info = future.result()
                chunk = np.array(data, dtype=np.float64)[:, :6]
                ohlcv_array[info["start"] : info["start"] + len(chunk)] = chunk
                total += len(chunk)
                print(length - total)

        df = self._create_dataframe(ohlcv_array, interval)

        return df

    def _create_dataframe(self, ohlcv_array: np.ndarray, interval: str) -> pd.DataFrame:
        columns = pd.Index(["timestamp", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(ohlcv_array, columns=columns)

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/Moscow")
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep="first")].asfreq(self.interval2freq(interval))
        return df

    @staticmethod
    def interval2freq(interval: str) -> str:
        if interval.endswith("m"):
            return interval[:-1] + "min"
        elif interval.endswith("h"):
            return interval[:-1] + "H"
        elif interval.endswith("d"):
            return interval[:-1] + "D"
        return interval

    @staticmethod
    def interval2ms(interval: str) -> int:
        num = int(interval[:-1])
        unit = interval[-1]

        multipliers = {
            "m": 60,
            "h": 60 * 60,
            "d": 24 * 60 * 60,
            "w": 7 * 24 * 60 * 60,
        }

        if unit not in multipliers:
            raise ValueError(f"Unsupported interval unit: {unit}")

        return multipliers[unit] * num * 1000
