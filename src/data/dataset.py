from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tsfs.core.task import TaskConfig

from ..scaler import ArcTan, Robust
from .download import ExchangeDownloader
from .utils import seq_split


class TradeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int, pred_len: int):
        self.df = df

        self.scaler_volume = ArcTan(0.3)
        self.scaler_price = Robust()

        self.seq_len = seq_len
        self.pred_len = pred_len

        self.data = df[["close", "high", "low", "volume"]].values

        smdf = df.rolling(window=3, min_periods=1).mean()
        self.target = smdf[["close", "high", "low"]].values
        self.feats = self._gen_time_features(df.index)

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len

        input_seq = self.data[s_begin:s_end]
        target_seq = self.target[s_end:r_end]

        price, pctx = self.scaler_price(input_seq[:, 0:3])
        volume, vctx = self.scaler_volume(input_seq[:, 3:4])

        x = np.concat([price, volume], axis=1)
        y, _ = self.scaler_price(target_seq, pctx)
        mark = self.feats[s_begin:r_end]

        context = np.concat([[pctx]], axis=0)

        return (
            torch.FloatTensor(x),
            torch.FloatTensor(y),
            torch.FloatTensor(mark),
            context,
        )

    def inverse(self, pred, context) -> np.ndarray:
        return self.scaler_price.unscale(pred, context[0])

    @staticmethod
    def _gen_time_features(dates: pd.Index) -> np.ndarray:
        months = dates.month.values
        days = dates.day.values
        weekdays = dates.weekday.values
        hours = dates.hour.values
        return np.column_stack([months, days, weekdays, hours])


class DatasetR:
    def __init__(self, path: Path, symbols: list[str], timeframe: str = "15m"):
        self.path = path
        self.symbols = symbols
        self.timeframe = timeframe

        self.active_loaders: list[DataLoader] = []

    def download(self, max_length: int = 4_000_000):
        exchange = ExchangeDownloader()

        for symbol in self.symbols:
            pairname = symbol + ":USDT"
            file = self.path / f"{pairname}-{self.timeframe}.pkl"

            if file.exists():
                continue

            df = exchange.download(pairname, self.timeframe, max_length)
            df.to_pickle(file)

    def loaders(
        self,
        task: TaskConfig,
        lengths: list[int],
        batch_size: int = 512,
        device: str = "cuda",
    ) -> list[DataLoader]:
        self.cleanup()

        datasets = [[] for i in range(len(lengths) + 1)]
        for symbol in self.symbols:
            df = pd.read_pickle(self.path / f"{symbol}:USDT-{self.timeframe}.pkl")
            dataset = TradeDataset(df, task.seq_len, task.pred_len)

            splits = seq_split(dataset, [len(dataset) - sum(lengths)] + lengths)

            for i, slice in enumerate(splits):
                datasets[i].append(slice)

        loaders = []
        for i, dt_list in enumerate(datasets):
            loader = DataLoader(
                ConcatDataset(dt_list),
                batch_size,
                shuffle=(i == 0),
                num_workers=12,
                pin_memory=device == "cuda",
                persistent_workers=True,
            )
            loaders.append(loader)

        self.active_loaders.extend(loaders)

        return loaders

    def cleanup(self):
        for loader in self.active_loaders:
            if loader and hasattr(loader, "_iterator"):
                del loader._iterator

    def __del__(self):
        self.cleanup()
