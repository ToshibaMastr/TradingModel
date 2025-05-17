import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import scale


class TradeDataset(Dataset):
    def __init__(
        self,
        path: str,
        seq_len: int,
        pred_len: int,
    ):
        df = pd.read_pickle(path)[-50_000:-10_000]
        df = df[~df.index.duplicated(keep="first")]

        self.df = df

        self.seq_len = seq_len
        self.pred_len = pred_len

        self.data = self.df[["close", "high", "low", "volume"]]

    def __len__(self):
        return len(self.df) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len

        input_seq = self.data[s_begin:s_end]
        target_seq = self.data[s_end:r_end]

        price, pmn, pmx = scale(input_seq[["close", "high", "low"]])
        volume, vmn, vmx = scale(input_seq[["volume"]])

        tprice, _, _ = scale(target_seq[["close", "high", "low"]], pmn, pmx)
        tvolume, _, _ = scale(target_seq[["volume"]], vmn, vmx)

        x = pd.concat([price, volume], axis=1).values
        y = pd.concat([tprice, tvolume], axis=1).values

        return (torch.FloatTensor(x), torch.FloatTensor(y))
