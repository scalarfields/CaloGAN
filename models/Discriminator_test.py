# delay the imports so running train.py -h doesn't take 5,234,807 years
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import h5py
import logging
import sys
from tqdm import tqdm
import numpy as np

from ops_pytorch import (minibatch_discriminator, minibatch_output_shape, Dense3D,
                    calculate_energy, scale)

from architectures_torch import build_Generator, build_Discriminator
class Discriminator(nn.Module):
    def __init__(self, sizes, nb_classes=1):
        super(Discriminator, self).__init__()
        self.nb_classes = nb_classes
        self.mbd = True
        self.sparsity = True
        self.sparsity_mbd = True

        self.layers = nn.ModuleList([
            build_Discriminator(mbd=self.mbd, sparsity=self.sparsity, sparsity_mbd=self.sparsity_mbd,sizes=sizes[2*i:(i+1)*2])
            for i in range(3)
        ])

#NOTE: con tutto true il fc ha bisogno di 1863, mentre nel caso false solo 1800
        self.fc = nn.Linear(1863, 1) 
        self.aux_fc = nn.Linear(1863, 1) if nb_classes > 1 else None

        # For minibatch discrimination
        self.dense3d_energy = Dense3D(first_dim=10, last_dim=10, input_shape=(3,1))

    def forward(self, inputs, input_energy):
        features = []
        energies = []

        # Extract features and energies from each calorimeter layer
        for i, input in enumerate(inputs):
            features.append(self.layers[i](input))
            energies.append(calculate_energy(input))

        # Concatenate features
        features = torch.cat(features, dim=1)
        energies = torch.cat(energies,dim=1)

        # Total energy across all rows
        total_energy = torch.sum(energies, dim=1, keepdim=True)

        # Minibatch discrimination on the raw energies
        K_energy = self.dense3d_energy(energies)
        mbd_energy = torch.tanh(minibatch_discriminator(K_energy))

        # Absolute deviation from input energy
        energy_well = torch.abs(total_energy - input_energy)
        
        # Binary y/n if it is over the input energy
        well_too_big = 10 * (energy_well > 5).float()

        # Concatenate all features
        p = torch.cat([
            features,
            scale(energies, 10),
            scale(total_energy, 100),
            energy_well,
            well_too_big,
            mbd_energy
        ], dim=1)

        fake = torch.sigmoid(self.fc(p))
        discriminator_outputs = [fake, total_energy]

        # Auxiliary output for ACGAN
        if self.nb_classes > 1:
            aux = torch.sigmoid(self.aux_fc(p))
            discriminator_outputs.append(aux)

        return discriminator_outputs
        


batch_size = 16
input_channels = 1

inputs = [3,96,12,12,12,6]

input_tensor_first = torch.randn(batch_size,input_channels,inputs[0], inputs[1])
input_tensor_second = torch.randn(batch_size,input_channels,inputs[2], inputs[3])
input_tensor_third = torch.randn(batch_size,input_channels,inputs[4], inputs[5])
input_energy = torch.randn(batch_size,1)

dataset = [input_tensor_first, input_tensor_second, input_tensor_third]
model = Discriminator(inputs,2)

out = model(dataset,input_energy)

print("final",out[0].shape,out[1].shape)