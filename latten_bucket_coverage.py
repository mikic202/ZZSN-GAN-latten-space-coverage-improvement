from collections import defaultdict
import numpy as np


def get_buckets(data: np.ndarray, projection: np.ndarray) -> set:
    """Maps a batch of features to a set of unique buckets via LSH."""
    result = data @ projection
    hashed = [tuple(int(x) for x in row) for row in (result > 0)]

    buckets = defaultdict(list)
    for i, row in enumerate(hashed):
        buckets[row].append(i)

    return set("".join(map(str, k)) for k in buckets.keys())


def compare_latten_coverage(
    original_data_latten, generated_data_latten, number_of_buckets
) -> float:
    projection = np.random.randn(original_data_latten.shape[1], number_of_buckets)
    original_coverage = get_buckets(original_data_latten, projection)
    generated_coverage = get_buckets(generated_data_latten, projection)

    return abs(len(original_coverage) - len(generated_coverage)) / len(
        original_coverage
    )
