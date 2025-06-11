import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .scaler import Scaler


class TradeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        scaler: Scaler,
        seq_len: int,
        pred_len: int,
    ):
        self.scaler = scaler

        self.seq_len = seq_len
        self.pred_len = pred_len

        self.data = df[["close", "high", "low"]].values
        self.feats = self._gen_time_features(df.index)

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len

        input_seq = self.data[s_begin:s_end]
        target_seq = self.data[s_end:r_end]

        x, pctx = self.scaler.scale(input_seq)
        y, _ = self.scaler.scale(target_seq, pctx)

        mark = self.feats[s_begin:r_end]

        context = np.concat([[pctx]], axis=0)

        return (
            torch.FloatTensor(x),
            torch.FloatTensor(y),
            torch.FloatTensor(mark),
            context,
        )

    def inverse(self, pred, context) -> np.ndarray:
        return self.scaler.unscale(pred, context[0])

    @staticmethod
    def _gen_time_features(dates: pd.Index) -> np.ndarray:
        months = dates.month.values
        days = dates.day.values
        weekdays = dates.weekday.values
        hours = dates.hour.values
        return np.column_stack([months, days, weekdays, hours])
