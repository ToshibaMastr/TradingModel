import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import scale


def autogena(df, start, end, seq_len):
    state = "min"
    index = start

    timestamp = df.index[0]

    while index + seq_len <= end:
        frame = df.iloc[index : index + seq_len]["close"]
        if state == "min":
            timestamp = frame.idxmin()
            df.loc[timestamp, "action"] = 1
            state = "max"
        else:
            timestamp = frame.idxmax()
            df.loc[timestamp, "action"] = 2
            state = "min"

        index = df.index.get_loc(timestamp) + 1


class TimeSeries(Dataset):
    def __init__(self, path: str, seq_len: int):
        df = pd.read_pickle(path)  # [-200_000:]
        df = df[~df.index.duplicated(keep="first")]

        self.seq_len = seq_len

        autogena(df, 0, len(df), seq_len)

        self.data = df[["close", "signal", "volume", "action"]]

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len

        input_seq = self.data[s_begin:s_end]
        target_seq = self.data[s_begin + 1 : s_end + 1]

        price, _, _ = scale(input_seq[["close"]])
        volume, _, _ = scale(input_seq[["volume"]])
        signal = input_seq[["signal"]]

        frame = pd.concat([price, signal, volume], axis=1).values

        x = input_seq["action"].values
        y = target_seq["action"].values

        return (torch.FloatTensor(frame), torch.LongTensor(x), torch.LongTensor(y))
