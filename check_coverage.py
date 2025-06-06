from models import Discriminator, Generator
from latten_bucket_coverage import get_buckets
import time
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def compare_latten_coverage(
    original_data_latten, generated_data_latten, number_of_buckets
) -> float:

    projection = np.random.randn(original_data_latten.shape[1], number_of_buckets)
    original_coverage = get_buckets(original_data_latten, PROJECTION)
    generated_coverage = get_buckets(generated_data_latten, PROJECTION)

    return abs(len(original_coverage) - len(generated_coverage)) / len(original_coverage)

# Reproducibility
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")


data = np.load('/net/tscratch/people/plgmichal1010/data/data_nonrandom_responses.npz')['arr_0']
data_cond = np.load('/net/tscratch/people/plgmichal1010/data/data_nonrandom_particles.npz')['arr_0']
data_cond = pd.DataFrame(data_cond, columns=['Energy','Vx','Vy','Vz','Px','Py','Pz','mass','charge'], dtype=np.float32)


# Preprocessing
data = np.log(data + 1).astype(np.float32)

x_train, y_train = data_cond, data

num_instances = x_train.values.shape[0]
indices = np.random.permutation(num_instances)


x = torch.tensor(x_train.values[:100]).to(device)
y = torch.tensor(y_train[:100]).to(device)

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, generator=torch.Generator().manual_seed(SEED))

generator = Generator(10, 9).to(device)
generator.load_state_dict(torch.load('generator.pth', weights_only=True))
generator.eval()

discriminator = Discriminator(9).to(device)
discriminator.load_state_dict(torch.load('discriminator.pth', weights_only=True))
discriminator.eval()


def binary_accuracy(preds, targets):
    return ((preds > 0.5) == targets).float().mean().item()

step = 0
seed = torch.randn(16, 10, generator=torch.Generator().manual_seed(SEED)).to(device)
seed_cond = y_train[20:36].to(device)


LATTEN_SPACE = 64
N=12
PROJECTION = torch.randn(LATTEN_SPACE, N, requires_grad=True).to(device)


latten_distances_thorugh_epochs = []
losses_thorugh_epochs = []
wassersteine_dist_list = []
lambda_gp = 10
start = time.time()
total_d_real_acc = 0
total_d_fake_acc = 0
total_g_acc = 0
latten_distances = []
losses = []
original_coverage_list = []
generated_coverage_list = []
for i, (cond, real_images) in enumerate(train_loader):
    batch_size = real_images.size(0)
    real_images = real_images.view(-1, 1, 44, 44).to(device)
    cond = cond.to(device)
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)
    noise = torch.randn(batch_size, 10, generator=torch.Generator().manual_seed(SEED)).to(device)
    fake_images = generator(noise, cond)

    outputs, latents_fake = discriminator(fake_images, cond)
    with torch.no_grad():
        _, latents_real = discriminator(real_images, cond)
        _, latents_fake = discriminator(fake_images, cond)

    original_coverage = get_buckets(latents_real, PROJECTION)
    generated_coverage = get_buckets(latents_fake, PROJECTION)
    original_coverage_list.append(original_coverage / N)
    generated_coverage_list.append(generated_coverage / N)




x_values = [(i + 1) * batch_size for i in range(len(original_coverage_list))]

plt.figure(figsize=(8, 6))
plt.plot(x_values, original_coverage_list, label='Original Coverage', marker='o')
plt.plot(x_values, generated_coverage_list, label='Generated Coverage', marker='s')
plt.xlabel('Number of Queries (examples)', fontsize=12)
plt.ylabel('Bucket Coverage', fontsize=12)
plt.title('Bucket Coverage vs. Number of Queries', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
plt.savefig('coverage_plot.png')