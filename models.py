# import matplotlib.pyplot as plt
from torch import nn
import torch

class Generator(nn.Module):
    def __init__(self, noise_dim, cond_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.05),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256 * 13 * 13),
            nn.BatchNorm1d(256 * 13 * 13),
            nn.Dropout(0.05),
            nn.LeakyReLU(0.1, inplace=True)
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
            nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(18 * 12 * 12 + cond_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, LATTEN_SPACE),
            nn.BatchNorm1d(LATTEN_SPACE),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1)
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