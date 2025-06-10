import numpy as np


class MinMax:
    def scale(self, data: np.ndarray, context: tuple[float, float] | None = None):
        mn, mx = context if context else (data.min(), data.max())
        scaled = -1 + (data - mn) * 2 / (mx - mn)
        return scaled, (mn, mx)

    def unscale(self, data: np.ndarray, context: tuple[float, float]):
        mn, mx = context
        return (data + 1) * (mx - mn) / 2 + mn
