#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.modules.utils import _pair


channel_softmax = nn.Softmax(dim=-1)

def scale(x, v):
    return x / v

""" I modified locally connected2d so that bias is a float variable: if it is set to zero no bias is introduced, if it is set to 
any other number, that will be the bias
I also made another change: provide the input size together with all the other parameters of the convolution. It will authomatically 
compute the output dimension and prepare all the weights.
For now I assume that kernel size, padding, and stride are the same in both h and w dimensions 
"""


class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, input_size, kernel_size, bias, stride=1, padding=0, dilation=1):
        super(LocallyConnected2d, self).__init__()
        output_size_0=int(1+ (input_size[0] + 2*padding - dilation*(kernel_size-1)-1)/stride)
        output_size_1=int(1+ (input_size[1] + 2*padding - dilation*(kernel_size-1)-1)/stride)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size_0, output_size_1, kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.full((1, out_channels, output_size_0, output_size_1), bias)
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
        # Sum in in_channel and kernel_size dims       
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])

        if self.bias is not None:
            out += self.bias 
        return out
    
""" in InpaintingAttention ho tolto la funzione bias initializer perché mi sembra di averla implementata per bene  dentro il locally connected,
ho aggiunto come argomento della classe InpaintingAttention la dimensione di input in modo da poterlo usare su input diversi come è richiesto 
nell'attention in train.py"""
### AGGIUNGERE ASSERT SULLE DIMENSIONI 
class InpaintingAttention(nn.Module):
    def __init__(self, constant=-10, input_size=[14,26]):
        super(InpaintingAttention, self).__init__()

        # Define zero padding layer
        self.pad = nn.ZeroPad2d((1, 1, 1, 1))
        

        # Define locally connected layer
        self.lcn = LocallyConnected2d(in_channels=2, out_channels=2, input_size=input_size, kernel_size=3, bias=constant, stride=1)

    def forward(self, primary, carryover):
        # Concatenate primary and carryover along the last dimension
        x = torch.cat((primary, carryover), dim=1)  #concatenate along channel dimension
        
        # Apply zero padding
        h = self.pad(x)
        
        # Apply locally connected layer
        h = self.lcn(h)

        # Compute weights using channel softmax
        weights = channel_softmax(h)
        
        # Compute the weighted sum
        weighted_sum = torch.sum(x * weights, dim=1, keepdim=True)

        return weighted_sum



""" Note that in PyTorch, constraints and regularizers are typically handled separately, 
and the above example assumes simple weight initialization without custom constraints or regularizers.
I managed to set initializers quite easily. Since there wasn't any predefined constraint on weights 
I didn't try to set any of them. In cas we'll see. 

I changed the settings so that input_shape is a compulsory argument because I was not able to manage **kwargs.

I deleted the get config method because since pytorch does not let you set the inizializers and constraints
from the outside in my opinion this method was useless
 """   
class Dense3D(nn.Module):
    """
    A 3D, trainable, dense tensor product layer.
    """

    def __init__(self, first_dim, last_dim, input_shape, activation=None, use_bias=True):
        super(Dense3D, self).__init__()
        self.first_dim = first_dim
        self.last_dim = last_dim
        self.activation = activation if activation else lambda x: x
        self.use_bias = use_bias
        self.input_shape = input_shape
        self.input_dim = self.input_shape[-1]
        # Apply constraints and regularizers (if any) after initialization
        
        # Initialize kernel (weights)
        assert len(input_shape) >= 2
        self.kernel = nn.Parameter(
            torch.randn(first_dim, self.input_dim, last_dim)
        )
        nn.init.xavier_uniform_(self.kernel)
   
        # Initialize bias if use_bias is True
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(first_dim, last_dim))
        else:
            self.bias = None
        
        nn.init.zeros_(self.bias)
        

    def forward(self, inputs):
        # Compute the output
        print(inputs.shape, self.kernel.shape)
        out = torch.tensordot(inputs, self.kernel, dims=([1],[1]))
        if self.use_bias:
            out += self.bias
        # Apply activation function
        if self.activation:
            out = self.activation(out)

        return out

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        return (input_shape[0], self.first_dim, self.last_dim)
    
    

def energy_error(requested_energy, received_energy):
    # Compute the difference
    difference = (received_energy - requested_energy) / 10000

    # Determine if the energy is over the requested amount
    over_energized = (difference > 0).float()
    print(over_energized)

    # Compute the penalties for too high and too low energy
    too_high = 100 * torch.abs(difference)
    too_low = 10 * torch.abs(difference)

    # Compute the final energy error
    return over_energized * too_high + (1 - over_energized) * too_low





def single_layer_energy_output_shape(input_shape):
    shape = list(input_shape)
    # assert len(shape) == 3
    return (shape[0], 1)
