from torch.utils.data import Subset


def scale(series, mn: float = None, mx: float = None):
    if mn is None or mx is None:
        val = series.values
        mn, mx = val.min(), val.max()
    scaled = -1 + (series - mn) * 2 / (mx - mn)
    return scaled, mn, mx


def unscale(series, mn: float, mx: float):
    original = (series + 1) * (mx - mn) / 2 + mn
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
