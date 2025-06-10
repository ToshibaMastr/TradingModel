import numpy as np


class Cache:
    def scale(self, data: np.ndarray, context: tuple[float, float] | None = None):
        if context is None:
            mean = data.mean()
        else:
            mean, _ = context

        scaled = (data / mean) - 1
        return scaled, (mean, mean)

    def unscale(self, data: np.ndarray, context: tuple[float, float]):
        mean, _ = context
        return (data + 1) * mean
