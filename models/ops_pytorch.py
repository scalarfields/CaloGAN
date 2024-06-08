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
In order to stack many of those layers I also return the output shape, which will be the input shape of the next layer
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
                torch.full((1, out_channels, output_size_0, output_size_1), bias, requires_grad=True)
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
        return out, out.shape
    
""" in InpaintingAttention ho tolto la funzione bias initializer perché mi sembra di averla implementata per bene  dentro il locally connected,
ho aggiunto come argomento della classe InpaintingAttention la dimensione di input"""

class InpaintingAttention(nn.Module):
    def __init__(self, constant=-10):
        super(InpaintingAttention, self).__init__()

        # Define zero padding layer
        self.pad = nn.ZeroPad2d((1, 1, 1, 1))

        # Define locally connected layer
        self.lcn = LocallyConnected2d(in_channels=6, out_channels=2, inp1, kernel_size=3, stride=0, bias=constant, stride=1) 
        #NOTA: inp1 deve essere la dimensione di quello che gli passo, ma va vista al momento

    def forward(self, primary, carryover):
        # Concatenate primary and carryover along the last dimension
        x = torch.cat((primary, carryover), dim=-1)  #concatenate along channel dimension

        # Apply zero padding
        h = self.pad(x)

        # Apply locally connected layer
        h = self.lcn(h)

        # Compute weights using channel softmax
        weights = channel_softmax(h)

        # Compute the weighted sum
        weighted_sum = torch.sum(x * weights, dim=-1, keepdim=True)

        return weighted_sum