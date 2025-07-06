from pathlib import Path

import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader
from tsfs.core.task import TaskConfig

from .dataset import TradeDataset
from .utils import seq_split


class DataManager:
    def __init__(self, path: Path, symbols: list[str], timeframe: str = "15m"):
        self.path = path
        self.symbols = symbols
        self.timeframe = timeframe

        self.active_loaders: list[DataLoader] = []

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
            df = pd.read_pickle(self.path / f"{symbol}USDT-{self.timeframe}.pkl")
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
