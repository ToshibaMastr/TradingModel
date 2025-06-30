from torch.utils.data import Dataset, Subset


def seq_split(dataset: Dataset, lengths: list[int]):
    subsets = []
    start = 0
    for length in lengths:
        subsets.append(Subset(dataset, range(start, start + length)))
        start += length

    return tuple(subsets)
