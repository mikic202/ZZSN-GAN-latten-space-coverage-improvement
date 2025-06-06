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
import logging
import csv
from latten_bucket_coverage import (
    get_buckets,
    jensen_shannon_loss,
    lsh_diversity_loss,
    one_dim_wasserstein_loss,
)
from scipy.stats import wasserstein_distance


task_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=f"task_{task_index}/output.log", encoding="utf-8", level=logging.DEBUG
)


def print_info(msg: str):
    print(msg)
    logger.info(msg)


current_user = pwd.getpwuid(os.getuid())[0]

print_info(f"SLURM Task index: {task_index}, Timestamp: {datetime.datetime.now()}")

# Reproducibility
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print_info(f"Using: {device}")

parameters = pd.read_csv("parameters.csv")
task_parameters = parameters.iloc[task_index]
print_info(f"Using parameters {task_parameters}")

# Load Data
data = np.load("/net/tscratch/people/plgmichal1010/data/data_nonrandom_responses.npz")[
    "arr_0"
]
data_cond = np.load(
    "/net/tscratch/people/plgmichal1010/data/data_nonrandom_particles.npz"
)["arr_0"]
data_cond = pd.DataFrame(
    data_cond,
    columns=["Energy", "Vx", "Vy", "Vz", "Px", "Py", "Pz", "mass", "charge"],
    dtype=np.float32,
)

# Preprocessing
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


print_info(f" len_train= {len(y_train)} len_test= {len(y_test)}")

batch_size = int(task_parameters["batch_s"])
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
)


batch_size = int(task_parameters["batch_s"])
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
)

# BUCKETS
LATTEN_SPACE = int(task_parameters["lat"])
N = int(task_parameters["N"])
PROJECTION = torch.randn(LATTEN_SPACE, N, requires_grad=True).to(device)


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
    mask1, mask2, mask3, mask4, mask5 = [
        torch.tensor(m).to(device) for m in [mask1, mask2, mask3, mask4, mask5]
    ]
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

                z = torch.randn(
                    end - i, 10, generator=torch.Generator().manual_seed(SEED)
                ).to(device)

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
        print_info(f"ws mean {ws.mean():.2f}  ")
        for n, score in enumerate(ws):
            print_info(f"ch{n+1} {score:.2f}  ")
        print_info("\n")
        torch.cuda.empty_cache()
        gc.collect()
    return ws.mean()


# Models
class Generator(nn.Module):
    def __init__(self, noise_dim, cond_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.05),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256 * 13 * 13),
            nn.BatchNorm1d(256 * 13 * 13),
            nn.Dropout(0.05),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.upsample = nn.Upsample(scale_factor=(2, 2))
        self.conv_layers = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.Dropout(0.05),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.Dropout(0.05),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Conv2d(256, 128, kernel_size=2),
            nn.BatchNorm2d(128),
            nn.Dropout(0.05),
            nn.LeakyReLU(0.15, inplace=True),
            nn.Conv2d(128, 1, kernel_size=2),
            nn.Sigmoid(),
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
            nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(18 * 12 * 12 + cond_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, LATTEN_SPACE),
            nn.BatchNorm1d(LATTEN_SPACE),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1),
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


# Instantiate models
generator = Generator(10, 9).to(device)
discriminator = Discriminator(9).to(device)

lr = task_parameters["lr"]

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)


def binary_accuracy(preds, targets):
    return ((preds > 0.5) == targets).float().mean().item()


step = 0
seed = torch.randn(16, 10, generator=torch.Generator().manual_seed(SEED)).to(device)
seed_cond = y_train[20:36].to(device)

alpha = task_parameters["alpha"]

EPOCHS = int(task_parameters["epochs"])

calculate_ws_ch(generator, x_test, y_test)


latten_distances_thorugh_epochs = []
losses_thorugh_epochs = []
wassersteine_dist_list = []
lambda_gp = 10
for epoch in range(EPOCHS):
    start = time.time()
    total_d_real_acc = 0
    total_d_fake_acc = 0
    total_g_acc = 0
    latten_distances = []
    losses = []
    for i, (cond, real_images) in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images.view(-1, 1, 44, 44).to(device)
        cond = cond.to(device)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        noise = torch.randn(
            batch_size, 10, generator=torch.Generator().manual_seed(SEED)
        ).to(device)
        fake_images = generator(noise, cond)

        outputs_real, latents_real = discriminator(real_images, cond)
        outputs_fake, _ = discriminator(fake_images.detach(), cond)

        d_real_acc = binary_accuracy(outputs_real, real_labels)
        d_fake_acc = binary_accuracy(outputs_fake, fake_labels)
        total_d_real_acc = (total_d_real_acc + d_real_acc) / 2
        total_d_fake_acc = (total_d_fake_acc + d_fake_acc) / 2

        loss_real = criterion(outputs_real, real_labels)
        loss_fake = criterion(outputs_fake, fake_labels)

        interpolated_samples = (
            alpha * real_images + (1 - alpha) * fake_images.detach()
        ).requires_grad_(True)
        d_interpolated, _ = discriminator(interpolated_samples, cond)

        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated_samples,
            grad_outputs=torch.ones_like(d_interpolated, device=device),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradient_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)

        loss_D = (
            outputs_fake.mean()
            - outputs_real.mean()
            + ((gradient_norm - 1) ** 2).mean() * lambda_gp
        )

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        outputs, latents_fake = discriminator(fake_images, cond)
        with torch.no_grad():
            _, latents_real = discriminator(real_images, cond)

        g_acc = binary_accuracy(outputs, real_labels)
        total_g_acc = (total_g_acc + g_acc) / 2

        latten_dist = one_dim_wasserstein_loss(latents_fake, latents_real)
        criterion_loss = 1 - outputs.mean()
        loss_G = criterion_loss + alpha * latten_dist

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        latten_distances.append(latten_dist.to("cpu").detach().numpy())
        losses.append(criterion_loss.to("cpu").detach().numpy())

    print(
        f"Time for epoch {epoch+1} is {time.time() - start:.2f} sec  [D real acc: {100 * total_d_real_acc:.2f}%] [D fake acc: {100 * total_d_fake_acc:.2f}%] [G acc: {100 * total_g_acc:.2f}%] [AVG lat dist: {sum(latten_distances)/len(latten_distances)}] [AVG loss: {sum(losses)/len(losses)}]"
    )
    latten_distances_thorugh_epochs.append(
        sum(latten_distances) / len(latten_distances)
    )
    losses_thorugh_epochs.append((sum(losses) / len(losses)))

    # if (epoch + 1) % 5 == 0:
    wassersteine_dist_list.append(calculate_ws_ch(generator, x_test, y_test))

with open(f"task_{task_index}/results.csv", "w") as file:
    csvwriter = csv.writer(file)
    for ld, loss, wass in zip(
        latten_distances_thorugh_epochs, losses_thorugh_epochs, wassersteine_dist_list
    ):
        csvwriter.writerow([ld, loss, wass])

torch.save(generator.state_dict(), f"task_{task_index}/models/generator.pth")
torch.save(discriminator.state_dict(), f"task_{task_index}/models/discriminator.pth")

print_info(f"Saved model with index {task_index}")
