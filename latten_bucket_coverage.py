import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
from collections import defaultdict


def get_buckets(data, projection):
    """Maps a batch of features to a set of unique buckets via LSH."""
    result = data @ projection
    return torch.nn.functional.sigmoid(result)

def get_desctrete_buckets(data, projection):
    result = data @ projection
    hashed = [tuple(int(x) for x in row) for row in (result > 0)]

    buckets = defaultdict(list)
    for i, row in enumerate(hashed):
        buckets[row].append(i)

    return set("".join(map(str, k)) for k in buckets.keys())


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

def soft_histogram(soft_hash, num_bins, sigma, projection):
    bin_centers = torch.linspace(0, 2 ** projection.shape[1] - 1, num_bins).to(
        soft_hash.device
    )
    dists = soft_hash.unsqueeze(1) - bin_centers.unsqueeze(0)
    weights = torch.exp(-(dists**2) / (2 * sigma**2))
    hist = weights.sum(dim=0)
    hist = hist / hist.sum()
    return hist


def lsh_diversity_loss(
    real_data, fake_data, projection, num_bins=128, sigma=5.0, alpha=30.0
):
    def get_soft_hash(data, projection):
        projected = data @ projection
        hash_bits = torch.sigmoid(projected)
        N = hash_bits.shape[1]
        powers = 2 ** torch.arange(N).float().to(data.device)
        return (hash_bits * powers).sum(dim=1)

    def estimate_occupied_bins(hist, alpha):
        return (1 - torch.exp(-alpha * hist)).sum()

    soft_hash_real = get_soft_hash(real_data, projection)
    soft_hash_fake = get_soft_hash(fake_data, projection)

    hist_real = soft_histogram(soft_hash_real, num_bins, sigma, projection)
    hist_fake = soft_histogram(soft_hash_fake, num_bins, sigma, projection)

    occ_real = estimate_occupied_bins(hist_real, alpha)
    occ_fake = estimate_occupied_bins(hist_fake, alpha)

    return torch.nn.functional.mse_loss(occ_fake, occ_real)

def one_dim_wasserstein_loss(buckets_real: torch.Tensor,
                             buckets_fake: torch.Tensor) -> torch.Tensor:
    proj_real = buckets_real.t()
    proj_fake = buckets_fake.t()

    real_sorted, _ = torch.sort(proj_real, dim=1)
    fake_sorted, _ = torch.sort(proj_fake, dim=1)

    wass_per_proj = torch.abs(real_sorted - fake_sorted).mean(dim=1)
    sw_dist = wass_per_proj.mean()

    return sw_dist