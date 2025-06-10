import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .scaler import Scaler


class TradeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        price_scaler: Scaler,
        volume_scaler: Scaler,
        seq_len: int,
        pred_len: int,
    ):
        self.price_scaler = price_scaler
        self.volume_scaler = volume_scaler

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

        price, pctx = self.price_scaler.scale(input_seq[:, 0:3])
        volume, vctx = self.volume_scaler.scale(input_seq[:, 3:])

        tprice, _ = self.price_scaler.scale(target_seq[:, 0:3], pctx)
        tvolume, _ = self.volume_scaler.scale(target_seq[:, 3:], vctx)

        x = np.concat([price, volume], axis=1)
        y = np.concat([tprice, tvolume], axis=1)
        mark = self.feats[s_begin:r_end]

        context = np.concat([[pctx], [vctx]], axis=0)

        return (
            torch.FloatTensor(x),
            torch.FloatTensor(y),
            torch.FloatTensor(mark),
            context,
        )

    def inverse(self, pred, context) -> np.ndarray:
        price = self.price_scaler.unscale(pred[:, 0:3], context[0])
        volume = self.volume_scaler.unscale(pred[:, 3:], context[1])
        return np.concat([price, volume], axis=1)

    @staticmethod
    def _gen_time_features(dates: pd.Index) -> np.ndarray:
        months = dates.month.values
        days = dates.day.values
        weekdays = dates.weekday.values
        hours = dates.hour.values
        return np.column_stack([months, days, weekdays, hours])
