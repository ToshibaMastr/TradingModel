import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import scale_robust


class TradeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int, pred_len: int):
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.data = df[["close", "high", "low", "volume"]].values
        self.feats = self._gen_time_features(df.index)

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len

        input_seq = self.data[s_begin:s_end]
        target_seq = self.data[s_end:r_end]

        price, pmn, pmx = scale_robust(input_seq[:, 0:3])
        volume, vmn, vmx = scale_robust(input_seq[:, 3:])

        tprice, _, _ = scale_robust(target_seq[:, 0:3], pmn, pmx)
        tvolume, _, _ = scale_robust(target_seq[:, 3:], vmn, vmx)

        x = np.concat([price, volume], axis=1)
        y = np.concat([tprice, tvolume], axis=1)
        mark = self.feats[s_begin:r_end]

        return (
            torch.FloatTensor(x),
            torch.FloatTensor(y),
            torch.FloatTensor(mark)
        )

    @staticmethod
    def _gen_time_features(dates: pd.Index) -> np.ndarray:
        months = dates.month.values
        days = dates.day.values
        weekdays = dates.weekday.values
        hours = dates.hour.values
        return np.column_stack([months, days, weekdays, hours])
