import time
import gc
import random
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from torch import nn, optim
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
import datetime
import pwd
import os

from latten_bucket_coverage import get_buckets, jensen_shannon_loss, kullback_leibler_div, get_desctrete_buckets
from scipy.stats import wasserstein_distance


task_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

current_user = pwd.getpwuid(os.getuid())[0]

print(f"SLURM Task index: {task_index}, Timestamp: {datetime.datetime.now()}")

# Reproducibility
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

parameters = pd.read_csv("parameters.csv")
task_parameters = parameters.iloc[task_index]
some_NN_param = int(task_parameters["NN_param"])
print(f"Using parameters {task_parameters}")

# Load Data
data = np.load(f'/net/people/plgrid/plgmchomans/data_nonrandom_responses.npy')
data_cond = np.load(f'/net/people/plgrid/plgmchomans/data_nonrandom_particles.npy')
data_cond = pd.DataFrame(data_cond, columns=['Energy','Vx','Vy','Vz','Px','Py','Pz','mass','charge'], dtype=np.float32)

data = np.log(data + 1).astype(np.float32)

x_train, y_train = data_cond, data

num_instances = x_train.values.shape[0]
indices = np.random.permutation(num_instances)

split_idx = int(0.9 * num_instances)
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]


x_test = torch.tensor(x_train.values[test_indices]).to(device)
y_test = torch.tensor(y_train[test_indices]).to(device)

x_train = torch.tensor(x_train.values[train_indices])
y_train = torch.tensor(y_train[train_indices])



batch_size = 100
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(SEED))


LATTEN_SPACE = 128
N=13
PROJECTION = torch.randn(LATTEN_SPACE, N, requires_grad=True).to(device)


# Add back calculate_ws_ch function
def get_channel_masks(shape):
    n, m = shape
    pattern = np.array([[0, 1], [1, 0]])
    mask = np.ones((n, m))
    for i in range(n):
        for j in range(m):
            mask[i, j] = pattern[i % 2, j % 2]
    mask5 = 1 - mask
    mid_row, mid_col = n // 2, m // 2
    mask1 = mask.copy()
    mask2 = mask.copy()
    mask3 = mask.copy()
    mask4 = mask.copy()
    mask4[mid_row:, :] = 0
    mask4[:, :mid_col] = 0
    mask2[:, :mid_col] = 0
    mask2[:mid_row, :] = 0
    mask3[mid_row:, :] = 0
    mask3[:, mid_col:] = 0
    mask1[:, mid_col:] = 0
    mask1[:mid_row, :] = 0
    return mask1, mask2, mask3, mask4, mask5

def sum_channels_parallel(data):
    mask1, mask2, mask3, mask4, mask5 = get_channel_masks(data.shape[1:])
    mask1, mask2, mask3, mask4, mask5 = [torch.tensor(m).to(device) for m in [mask1, mask2, mask3, mask4, mask5]]
    ch1 = (data * mask1).sum(dim=[1, 2])
    ch2 = (data * mask2).sum(dim=[1, 2])
    ch3 = (data * mask3).sum(dim=[1, 2])
    ch4 = (data * mask4).sum(dim=[1, 2])
    ch5 = (data * mask5).sum(dim=[1, 2])
    return torch.stack([ch1, ch2, ch3, ch4, ch5], dim=1)

def calculate_ws_ch(generator, y_test, x_test, n_calc=5, batch_size=256):
    with torch.no_grad():
        org = torch.exp(x_test) - 1
        ch_org = org.view(-1, 44, 44)
        ch_org = sum_channels_parallel(ch_org).cpu().numpy()

        ws = np.zeros(5)
        n_samples = x_test.size(0)

        for _ in range(n_calc):
            ch_gen_list = []

            for i in range(0, n_samples, batch_size):
                end = min(i + batch_size, n_samples)
                z = torch.randn(end - i, 10, generator=torch.Generator().manual_seed(SEED)).to(device)
                y_batch = y_test[i:end]
                fake = generator(z, y_batch)
                fake = torch.exp(fake) - 1
                fake = fake.view(-1, 44, 44)
                ch_fake = sum_channels_parallel(fake).cpu().numpy()
                ch_gen_list.append(ch_fake)

            ch_gen = np.concatenate(ch_gen_list, axis=0)

            for i in range(5):
                ws[i] += wasserstein_distance(ch_org[:, i], ch_gen[:, i])

        ws /= n_calc
        print("ws mean", f"{ws.mean():.2f}", end=" ")
        for n, score in enumerate(ws):
            print(f"ch{n+1} {score:.2f}", end=" ")
        print()
        torch.cuda.empty_cache()
        gc.collect()



class Generator(nn.Module):
    def __init__(self, noise_dim, cond_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256 * 13 * 13),
            nn.BatchNorm1d(256 * 13 * 13),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=(2, 2))
        self.conv_layers = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(256, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Conv2d(128, 128, kernel_size=2),
            nn.BatchNorm2d(128),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.15, inplace=True),
            nn.Conv2d(128, 1, kernel_size=2),
            nn.Sigmoid()
        )

    def forward(self, noise, cond):
        x = torch.cat((noise, cond), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 256, 13, 13)
        x = self.upsample(x)
        x = self.conv_layers(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, cond_dim):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(9 * 12 * 12 + cond_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, LATTEN_SPACE),
            nn.BatchNorm1d(LATTEN_SPACE),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2)
        )
        self.fc3 = nn.Linear(LATTEN_SPACE, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, cond):
        x = self.conv_layers(img)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, cond), dim=1)
        x = self.fc1(x)
        latent = self.fc2(x)
        out = self.fc3(latent)
        out = self.sigmoid(out)
        return out, latent




generator = Generator(10, 9).to(device)
discriminator = Discriminator(9).to(device)
generator.load_state_dict(torch.load(f"/net/people/plgrid/plgmchomans/test_10_really_good/task_0/models/generator.pth", map_location=device))
discriminator.load_state_dict(torch.load(f"/net/people/plgrid/plgmchomans/test_10_really_good/task_0/models/discriminator.pth", map_location=device))

criterion = nn.BCELoss()

step = 0

PROJECTION = torch.randn(LATTEN_SPACE, N, requires_grad=True).to(device)
unique_buckets = set()
gen_unique_buckets = set()
increase_of_bucket_coverage = []
increase_of_samples = []
increase_of_gen_bucket_coverage = []
output = None
agregated_batch_size = 0
for cond, real_images in train_loader:
    batch_size = real_images.size(0)
    agregated_batch_size += batch_size
    real_images = real_images.view(-1, 1, 44, 44).to(device)
    noise = torch.randn(batch_size, 10).to(device)
    cond = cond.to(device)
    fake_images = generator(noise, cond)
    if output is None:
        output = fake_images.detach().cpu()
    elif agregated_batch_size <= 1000:
        output = torch.concat((output, fake_images.detach().cpu()))

    _, latents_real = discriminator(real_images, cond)
    _, lattens_fake = discriminator(fake_images, cond)

    gen_unique_buckets.update(get_desctrete_buckets(lattens_fake, PROJECTION))
    unique_buckets.update(get_desctrete_buckets(latents_real, PROJECTION))
    increase_of_bucket_coverage.append(len(unique_buckets))
    increase_of_gen_bucket_coverage.append(len(gen_unique_buckets))
    increase_of_samples.append(agregated_batch_size)

pd.DataFrame({"coverage": increase_of_bucket_coverage, "samples": increase_of_samples, "gen_coverage": increase_of_gen_bucket_coverage}).to_csv(f"Bucket_cov_N{N}.csv")
torch.save(output, "generated_images.pth")

print(f"Train set covers {len(unique_buckets)*100/2**N:.2f}% of latten space")

calculate_ws_ch(generator, x_train, y_test)

print(f"Saved model with index {task_index}")