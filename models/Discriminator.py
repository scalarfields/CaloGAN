import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import numpy as np

#from ops import (minibatch_discriminator, minibatch_output_shape,
#                 Dense3D, sparsity_level, sparsity_output_shape)

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
    
#TODO: Verify that strides actually work
#TODO: Convert to Martina's locally conneced convention
#TODO: vary inputsize 
class Discriminator(nn.Module):
    def __init__(self, mbd=False, sparsity=False, sparsity_mbd=False):
        super(Discriminator, self).__init__()
        self.mbd = mbd
        self.sparsity = sparsity
        self.sparsity_mbd = sparsity_mbd

        self.conv1 = nn.Conv2d(1, 64, kernel_size=2, padding='same')
        self.lrelu = nn.LeakyReLU()

        self.zero_pad = nn.ZeroPad2d(1)
        self.local_conv1 = LocallyConnected2d(64, 16, output_size=(10, 5), kernel_size=3, stride=(1,2))
        self.batch_norm1 = nn.BatchNorm2d(16)

        self.local_conv2 = LocallyConnected2d(16, 8, output_size=(11, 6), kernel_size=2)
        self.batch_norm2 = nn.BatchNorm2d(8)

        self.local_conv3 = LocallyConnected2d(8, 8, output_size=(12, 4), kernel_size=2, stride=(1,2))
        self.batch_norm3 = nn.BatchNorm2d(8)

        self.flatten = nn.Flatten()

        """ if self.mbd or self.sparsity or self.sparsity_mbd:
            self.minibatch_featurizer = minibatch_discriminator
            self.dense3d = Dense3D(10, 10)
            self.activation_tanh = nn.Tanh()
            self.sparsity_detector = sparsity_level """

    def forward(self, image):
        x = image.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.lrelu(x)

        x = self.zero_pad(x)

        x = self.local_conv1(x)
        x = self.lrelu(x)
        x = self.batch_norm1(x)

        x = self.zero_pad(x)
        print(x.shape,"after loc2")
        x = self.local_conv2(x)
        x = self.lrelu(x)
        x = self.batch_norm2(x)

        x = self.zero_pad(x)
        x = self.local_conv3(x)
        x = self.lrelu(x)
        x = self.batch_norm3(x)

        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
        """ if self.mbd or self.sparsity or self.sparsity_mbd:
            features = [x]
            if self.mbd:
                K_x = self.dense3d(x)
                features.append(self.activation_tanh(self.minibatch_featurizer(K_x)))

            if self.sparsity or self.sparsity_mbd:
                empirical_sparsity = self.sparsity_detector(image)
                if self.sparsity:
                    features.append(empirical_sparsity)
                if self.sparsity_mbd:
                    K_sparsity = self.dense3d(empirical_sparsity)
                    features.append(self.activation_tanh(self.minibatch_featurizer(K_sparsity)))

            return torch.cat(features, dim=1)
        else: """
        return x



# Creare un'istanza del modello
discriminator = Discriminator(mbd=False, sparsity=False, sparsity_mbd=False)

# Generare input casuale
batch_size = 16
input_channels = 1
height = 10
width = 10
input_tensor = torch.randn(batch_size,height, width,input_channels)

# Passare l'input attraverso il modello
output_tensor = discriminator(input_tensor)

# Verificare la forma dell'output
expected_output_shape = (batch_size, (height+6)*(width/2-2)*4)
#assert output_tensor.shape == expected_output_shape, f"Expected output shape {expected_output_shape}, got {output_tensor.shape}"

# Verificare se tutti gli elementi dell'output sono finiti
assert torch.isfinite(output_tensor).all(), "Output contains non-finite elements"

print("Discriminator model tests passed successfully!")
