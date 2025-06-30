import numpy as np


class ArcTan:
    def __init__(self, sensitivity: float = 1.0):
        self.sensitivity = sensitivity

    def __call__(self, data: np.ndarray, context: tuple[float, float] | None = None):
        if context is None:
            mean = data.mean()
        else:
            mean, _ = context

        normalized = (data - mean) / mean
        scaled = np.arctan(normalized * self.sensitivity) * (2 / np.pi)
        return scaled, (mean, mean)

    def unscale(self, data: np.ndarray, context: tuple[float, float]):
        mean, _ = context
        arctan_reversed = data * (np.pi / 2)
        normalized = np.tan(arctan_reversed) / self.sensitivity
        return (normalized * mean) + mean
