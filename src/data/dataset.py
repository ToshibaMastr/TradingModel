from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from tsfs.core.task import TaskConfig

from ..scaler import ArcTan, Robust
from .utils import seq_split
from .download import ExchangeDownloader

class TradeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        pred_len: int,
    ):
        self.scaler_volume = ArcTan()
        self.scaler_price = Robust()

        self.seq_len = seq_len
        self.pred_len = pred_len

        self.data = df[["close", "high", "low", "volume"]].values
        self.target = df[["close", "high", "low"]].values
        self.feats = self._gen_time_features(df.index)

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len

        input_seq = self.data[s_begin:s_end]
        target_seq = self.target[s_end:r_end]

        price, pctx = self.scaler_price.scale(input_seq[:, 0:3])
        volume, vctx = self.scaler_volume.scale(input_seq[:, 3:])

        y, _ = self.scaler_price.scale(target_seq, pctx)

        x = np.concat([price, volume], axis=1)
        mark = self.feats[s_begin:r_end]

        context = np.concat([[pctx]], axis=0)

        return (
            torch.FloatTensor(x),
            torch.FloatTensor(y),
            torch.FloatTensor(mark),
            context
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
    def __init__(self, path: Path, symbols: list[str], timeframe: str = "15m", val_size: int = 4000):
        self.path = path
        self.symbols = symbols
        self.timeframe = timeframe
        self.val_size = val_size

        self.val_loader: Optional[DataLoader] = None
        self.train_loader: Optional[DataLoader] = None

    def download(self, max_length: int = 4_000_000):
        exchange = ExchangeDownloader()

        for symbol in self.symbols:
            pair = symbol + "/USDT:USDT"

            pairname = pair.split(":")[0].replace("/", ":")
            file = self.path / f"{pairname}-{self.timeframe}.pkl"

            if file.exists():
                continue

            df = exchange.download(pair, self.timeframe, max_length)
            df.to_pickle(file)

    def loaders(
        self, task: TaskConfig, batch_size: int = 512, device: str = "cuda"
    ) -> tuple[DataLoader, DataLoader]:
        self.cleanup()

        datasets = []
        val_datasets = []
        for symbol in self.symbols:
            df = pd.read_pickle(self.path / f"{symbol}:USDT-{self.timeframe}.pkl") # [-10_000:]
            dataset = TradeDataset(df, task.seq_len, task.pred_len)
            train, val = seq_split(dataset, [len(dataset) - self.val_size, self.val_size])
            datasets.append(train)
            val_datasets.append(val)

        train_loader = DataLoader(
            ConcatDataset(datasets),
            batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=device == "cuda",
            persistent_workers=True,
        )
        val_loader = DataLoader(
            ConcatDataset(val_datasets),
            batch_size,
            num_workers=12,
            pin_memory=device == "cuda",
            persistent_workers=True,
        )

        self.val_loader = val_loader
        self.train_loader = train_loader

        return train_loader, val_loader

    def cleanup(self):
        for loader in [self.val_loader, self.train_loader]:
            if loader:
                del loader._iterator

    def __del__(self):
        self.cleanup()
