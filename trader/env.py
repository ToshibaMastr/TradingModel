import pandas as pd
import torch

from ..utils import scale


class TradingEnv:
    def __init__(self, path, initial_cash=1000):
        data = pd.read_pickle(path)[["close", "signal", "volume"]][-1200:]
        data = data[~data.index.duplicated(keep="first")]
        data["action"] = 0
        self.data = data.reset_index(drop=True)

        self.initial_cash = initial_cash
        self.reset()

    def reset(self):
        self.current_step = 1000
        self.position = 0
        self.entry_price = 0
        self.cash = self.initial_cash
        self.active = True

    def get_state(self):
        frame = self.data[self.current_step - 32 : self.current_step]
        price, _, _ = scale(frame[["close"]])
        volume, _, _ = scale(frame[["volume"]])
        signal = frame[["signal"]]
        action = frame[["action"]].values
        state = pd.concat([price, volume, signal], axis=1).values
        return torch.FloatTensor(state[-1]), torch.LongTensor(action[-1])

    def step(self, action):
        reward = 0

        timestamp = self.current_step
        current_price = self.data.loc[timestamp, "close"]

        self.data.loc[timestamp, "action"] = action

        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price
        elif action == 2 and self.position == 1:
            reward = current_price - self.entry_price
            self.cash += reward
            self.position = 0
            self.entry_price = 0
        elif action == 0 and self.position == 1:
            reward = (current_price - self.entry_price) * 0.1

        self.current_step += 1
        if timestamp >= len(self.data) - 1:
            self.active = False

        return reward
