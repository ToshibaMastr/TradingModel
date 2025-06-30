import numpy as np


class Robust:
    def __call__(self, data: np.ndarray, context: tuple[float, float] | None = None):
        if context is None:
            median = np.median(data)
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
        else:
            iqr, median = context

        scaled = (data - median) / iqr
        return scaled, (iqr, median)

    def unscale(self, data: np.ndarray, context: tuple[float, float]):
        iqr, median = context
        return (data * iqr) + median
