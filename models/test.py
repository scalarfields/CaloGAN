import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import numpy as np

# Custom LocallyConnected2d Layer
class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride=1, bias=True):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out
class Generator(nn.Module):
    def __init__(self, latent_dim, nb_rows, nb_cols):
        super(Generator, self).__init__()
        self.fc = nn.Linear(latent_dim, (nb_rows + 2) * (nb_cols + 2) * 36)
        self.reshape_size = (nb_rows + 2, nb_cols + 2, 36)

        self.conv1 = nn.Conv2d(36, 16, kernel_size=2,padding='same')
        self.lrelu = nn.LeakyReLU()
        self.batch_norm = nn.BatchNorm2d(16)

        # Using custom LocallyConnected2d layers
        self.local_conv1 = LocallyConnected2d(16, 6, output_size=(nb_rows+1, nb_cols+1), kernel_size=2)
        self.local_conv2 = LocallyConnected2d(6, 1, output_size=(nb_rows, nb_cols), kernel_size=2)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, *self.reshape_size)  # (batch_size, nb_rows+2, nb_cols+2, 36)
        
        x = x.permute(0, 3, 1, 2)  # (batch_size, 36, nb_rows+2, nb_cols+2)
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.batch_norm(x)
        x = self.local_conv1(x)
        x = self.lrelu(x)
        
        x = self.local_conv2(x)
        print(x.shape)
        return x


latent_dim = 100
nb_rows = 3
nb_cols = 96
model = Generator(latent_dim, nb_rows, nb_cols)

# Creare un input fittizio
dimensione_batch = 16
rumore = torch.randn(dimensione_batch, latent_dim)  # Batch di 16 vettori con dimensione latent_dim

# Generare immagini fake
immagini_fake = model(rumore)

# Verificare la forma delle immagini generate
assert immagini_fake.shape == (dimensione_batch, 1, nb_rows, nb_cols), "Le immagini generate hanno una forma incorretta"

# Verificare se tutti gli elementi sono finiti
assert torch.isfinite(immagini_fake).all(), "Le immagini generate contengono elementi non finiti"

print("Test del generatore superati con successo!")