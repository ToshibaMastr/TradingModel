import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import normalize


class TradeDataset(Dataset):
    def __init__(self, path: str, seq_len: int, label_len: int, pred_len: int):
        df = pd.read_pickle(path)[-500 * 512 :]
        df.drop(columns=["open"], inplace=True, errors="ignore")
        df = df[~df.index.duplicated(keep="first")]

        df_norm, _, _ = normalize(df, seq_len + label_len + pred_len)
        self.df = df_norm

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.time_feats = self._gen_time_features(self.df.index)

    def __len__(self):
        return len(self.df) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        x = self.df.iloc[s_begin:s_end].values
        y = self.df.iloc[r_begin:r_end].values
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
