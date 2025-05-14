def scale(series, mn: float = None, mx: float = None):
    if mn is None or mx is None:
        val = series.values
        mn, mx = val.min(), val.max()
    scaled = -1 + (series - mn) * 2 / (mx - mn)
    return scaled, mn, mx


def unscale(series, mn: float, mx: float):
    original = (series + 1) * (mx - mn) / 2 + mn
    return original
