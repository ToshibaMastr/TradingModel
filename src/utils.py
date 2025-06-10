from torch.utils.data import Subset
import numpy as np

def scale(series, mn = None, mx = None):
    if mn is None or mx is None:
        val = series.values
        mn, mx = val.min(), val.max()
    scaled = -1 + (series - mn) * 2 / (mx - mn)
    return scaled, mn, mx


def unscale(series, mn: float, mx: float):
    original = (series + 1) * (mx - mn) / 2 + mn
    return original


def scale_robust(val, iqr=None, median=None):
    if iqr is None or median is None:
        median = np.median(val)
        q1 = np.percentile(val, 25)
        q3 = np.percentile(val, 75)
        iqr = q3 - q1
    scaled = (val - median) / iqr
    return scaled, iqr, median

def unscale_robust(val, iqr, median):
    original = (val * iqr) + median
    return original


def scale_arctan(val, sensitivity=1.0, mean=None):
    if mean is None:
        mean = val.mean()
    normalized = (val - mean) / mean
    scaled = np.arctan(normalized * sensitivity) * (2 / np.pi)
    return scaled, mean, sensitivity


def unscale_arctan(val, mean, sensitivity):
    arctan_reversed = val * (np.pi / 2)
    normalized = np.tan(arctan_reversed) / sensitivity
    original = (normalized * mean) + mean
    return original


def seq_split(dataset, lengths):
    total = len(dataset)
    indices = list(range(total))
    subsets = []
    start = 0
    for length in lengths:
        end = start + length
        subsets.append(Subset(dataset, indices[start:end]))
        start = end
    return tuple(subsets)
