import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import scale


class TradeDataset(Dataset):
    def __init__(self, path: str, seq_len: int, label_len: int, pred_len: int):
        df = pd.read_pickle(path)[-2_000:]
        df = df[~df.index.duplicated(keep="first")]
        df["target"] = df["close"].rolling(window=5, min_periods=1).mean()

        self.data = df

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.time_feats = self._gen_time_features(self.data.index)

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        input_seq = self.data[s_begin:s_end]
        target_seq = self.data[r_begin:r_end]

        price, pmn, pmx = scale(input_seq[["close", "high", "low"]])
        volume, vmn, vmx = scale(input_seq[["volume"]])

        tprice, _, _ = scale(target_seq[["close", "high", "low"]], pmn, pmx)
        tvolume, _, _ = scale(target_seq[["volume"]], vmn, vmx)

        x = pd.concat([price, volume], axis=1).values
        y = pd.concat([tprice, tvolume], axis=1).values

        x_mark = self.time_feats[s_begin:s_end]
        y_mark = self.time_feats[r_begin:r_end]

        return (
            torch.FloatTensor(x),
            torch.FloatTensor(y),
            torch.FloatTensor(x_mark),
            torch.FloatTensor(y_mark),
        )

    @staticmethod
    def _gen_time_features(dates: pd.DatetimeIndex) -> np.ndarray:
        months = dates.month.values
        days = dates.day.values
        weekdays = dates.weekday.values
        hours = dates.hour.values
        return np.column_stack([months, days, weekdays, hours])
