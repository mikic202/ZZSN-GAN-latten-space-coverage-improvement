import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance


def get_buckets(data, projection):
    """Maps a batch of features to a set of unique buckets via LSH."""
    result = data @ projection
    return torch.nn.functional.sigmoid(result)


def kullback_leibler_div(data, data_to_compare):
    return nn.functional.cross_entropy(
        data, data_to_compare
    ) - nn.functional.cross_entropy(data, data)


def jensen_shannon_loss(data, data_to_compare):
    return kullback_leibler_div(
        data, (data + data_to_compare) / 2
    ) + kullback_leibler_div(data_to_compare, (data + data_to_compare) / 2)


def wasserstein_loss(data, data_to_compare):
    return wasserstein_distance(data, data_to_compare)


def gaussian_kernel(x, y, sigma=1.0):
    x_norm = (x**2).sum(dim=1).view(-1, 1)
    y_norm = (y**2).sum(dim=1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    return torch.exp(-dist / (2 * sigma**2))


def maximum_mean_discrepancy(x, y, sigma=1.0):
    return (
        gaussian_kernel(x, x, sigma).mean()
        + gaussian_kernel(y, y, sigma).mean()
        - 2 * gaussian_kernel(x, y, sigma).mean()
    )
