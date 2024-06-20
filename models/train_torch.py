#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
file: train.py
description: main training script for [arXiv/1705.02355]
author: Luke de Oliveira (lukedeo@manifold.ai), 
        Michela Paganini (michela.paganini@yale.edu)
"""

from __future__ import print_function

import argparse
from collections import defaultdict
import logging


import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import sys
import yaml
import time


if __name__ == '__main__':
    logger = logging.getLogger(
        '%s.%s' % (
            __package__, os.path.splitext(os.path.split(__file__)[-1])[0]
        )
    )
    logger.setLevel(logging.INFO)
else:
    logger = logging.getLogger(__name__)


def bit_flip(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x


def get_parser():
    parser = argparse.ArgumentParser(
        description='Run CalGAN training. '
        'Sensible defaults come from [arXiv/1511.06434]',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--nb-epochs', action='store', type=int, default=50,
                        help='Number of epochs to train for.')

    parser.add_argument('--batch-size', action='store', type=int, default=256,
                        help='batch size per update')

    parser.add_argument('--latent-size', action='store', type=int, default=1024,
                        help='size of random N(0, 1) latent space to sample')

    parser.add_argument('--disc-lr', action='store', type=float, default=2e-5,
                        help='Adam learning rate for discriminator')

    parser.add_argument('--gen-lr', action='store', type=float, default=2e-4,
                        help='Adam learning rate for generator')

    parser.add_argument('--adam-beta', action='store', type=float, default=0.5,
                        help='Adam beta_1 parameter')

    parser.add_argument('--prog-bar', action='store_true',
                        help='Whether or not to use a progress bar')

    parser.add_argument('--no-attn', action='store_true',
                        help='Whether to turn off the layer to layer attn.')

    parser.add_argument('--debug', action='store_true',
                        help='Whether to run debug level logging')

    parser.add_argument('--d-pfx', action='store',
                        default='params_discriminator_epoch_',
                        help='Default prefix for discriminator network weights')

    parser.add_argument('--g-pfx', action='store',
                        default='params_generator_epoch_',
                        help='Default prefix for generator network weights')

    parser.add_argument('dataset', action='store', type=str,
                        help='yaml file with particles and HDF5 paths (see '
                        'github.com/hep-lbdl/CaloGAN/blob/master/models/'
                        'particles.yaml)')

    return parser


if __name__ == '__main__':

    parser = get_parser()
    parse_args = parser.parse_args()

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

    from ops_pytorch import (minibatch_discriminator, minibatch_output_shape, Dense3D,
                     calculate_energy, scale, InpaintingAttention)

    from architectures_torch import build_Generator, build_Discriminator

    # batch, latent size, and whether or not to be verbose with a progress bar

    if parse_args.debug:
        logger.setLevel(logging.DEBUG)

    # set up all the logging stuff
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s'
        '[%(levelname)s]: %(message)s'
    )
    hander = logging.StreamHandler(sys.stdout)
    hander.setFormatter(formatter)
    logger.addHandler(hander)

    nb_epochs = parse_args.nb_epochs
    batch_size = parse_args.batch_size
    latent_size = parse_args.latent_size
    verbose = parse_args.prog_bar
    no_attn = parse_args.no_attn

    disc_lr = parse_args.disc_lr
    gen_lr = parse_args.gen_lr
    adam_beta_1 = parse_args.adam_beta

    yaml_file = parse_args.dataset

    logger.debug('parameter configuration:')

    logger.debug('number of epochs = {}'.format(nb_epochs))
    logger.debug('batch size = {}'.format(batch_size))
    logger.debug('latent size = {}'.format(latent_size))
    logger.debug('progress bar enabled = {}'.format(verbose))
    logger.debug('Using attention = {}'.format(no_attn == False))
    logger.debug('discriminator learning rate = {}'.format(disc_lr))
    logger.debug('generator learning rate = {}'.format(gen_lr))
    logger.debug('Adam $\beta_1$ parameter = {}'.format(adam_beta_1))
    logger.debug('Will read YAML spec from {}'.format(yaml_file))

    # read in data file spec from YAML file
    with open(yaml_file, 'r') as stream:
        try:
            s = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)
            raise exc
    nb_classes = len(s.keys())
    logger.info('{} particle types found.'.format(nb_classes))
    for name, pth in s.items():
        logger.debug('class {} <= {}'.format(name, pth))

    def _load_data(particle, datafile):


        d = h5py.File(datafile, 'r')

        # make our calo images channels-last
        first = np.expand_dims(d['layer_0'][:], -1)
        second = np.expand_dims(d['layer_1'][:], -1)
        third = np.expand_dims(d['layer_2'][:], -1)
        # convert to MeV
        energy = d['energy'][:].reshape(-1, 1) * 1000

        sizes = [
            first.shape[1], first.shape[2],
            second.shape[1], second.shape[2],
            third.shape[1], third.shape[2]
        ]

        y = [particle] * first.shape[0]

        d.close()

        return first, second, third, y, energy, sizes

    logger.debug('loading data from {} files'.format(nb_classes))

    first, second, third, y, energy, sizes = [
        np.concatenate(t) for t in [
            a for a in zip(*[_load_data(p, f) for p, f in s.items()])
        ]
    ]

    # TO-DO: check that all sizes match, so I could be taking any of them
    sizes = sizes[:6].tolist()

    # scale the energy depositions by 1000 to convert MeV => GeV
    first, second, third, energy = [
        (X.astype(np.float32) / 1000)
        for X in [first, second, third, energy]
    ]

    le = LabelEncoder()
    y = le.fit_transform(y)


    # Convert to PyTorch tensors
    
    first = torch.tensor(first, dtype=torch.float32).permute(0, 3, 1, 2)
    second = torch.tensor(second, dtype=torch.float32).permute(0, 3, 1, 2)
    third = torch.tensor(third, dtype=torch.float32).permute(0, 3, 1, 2)
    y = torch.tensor(y, dtype=torch.long)
    energy = torch.tensor(energy, dtype=torch.float32, requires_grad=True)

    # Dataset and DataLoader
    dataset = TensorDataset(first, second, third, energy,y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    logger.info('Building discriminator')
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

    class Generator(nn.Module):
        def __init__(self, latent_size, no_attn, nb_classes=1):
            super(Generator, self).__init__()
            self.latent_size = latent_size
            self.nb_classes = nb_classes
            self.no_attn = no_attn

            # Embedding layer
            if nb_classes > 1:
                self.embedding = nn.Embedding(nb_classes, latent_size)
                self.flatten = nn.Flatten()


            # Define generator layers
            self.gen_layer0 = build_Generator(latent_size, 3, 96)
            self.gen_layer1 = build_Generator(latent_size, 12, 12)
            self.gen_layer2 = build_Generator(latent_size, 12, 6)

            if not no_attn:
                self.attn_layer1=InpaintingAttention(constant=-10.0, input_size=[14,14])
                self.attn_layer2=InpaintingAttention(constant=-10.0, input_size=[14,8])


        def forward(self, generator_inputs, image_class=None):
            latent=generator_inputs[0]
            input_energy=generator_inputs[1]
            if self.nb_classes > 1 and image_class is not None:
                emb = self.embedding(image_class)
                emb = self.flatten(emb)
                hc = latent * emb
                h = hc * scale(input_energy, 100)
            else:
                h = latent * scale(input_energy, 100).shape[1]

            img_layer0 = self.gen_layer0(h)
            img_layer1 = self.gen_layer1(h)
            img_layer2 = self.gen_layer2(h)

            if not no_attn:
                # resizes from (3, 96) => (12, 12)
                zero2one = nn.AvgPool2d(kernel_size=(1, 8))(
                    nn.Upsample(scale_factor=(4, 1), mode='nearest')(img_layer0))
                img_layer1 = self.attn_layer1(img_layer1, zero2one)
                
                # resizes from (12, 12) => (12, 6)
                one2two = nn.AvgPool2d(kernel_size=(1, 2))(img_layer1)
                img_layer2 = self.attn_layer2(img_layer2, one2two)
        

            return [F.relu(img_layer0), F.relu(img_layer1), F.relu(img_layer2)]
        
    discriminator = Discriminator(sizes)
    generator=Generator(latent_size, False)

    logger.info('commencing training')

    opt_g = torch.optim.Adam(params=generator.parameters(), lr=gen_lr, weight_decay=1e-5)
    opt_d = torch.optim.Adam(params=discriminator.parameters(), lr=disc_lr, weight_decay=1e-5)

    bce_loss=nn.BCELoss()
    mae_loss=nn.L1Loss()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    discriminator.to(device)

    disc_loss = []
    gen_loss = []
        
    ones = torch.ones(batch_size,device=device)
    zeros = torch.zeros(batch_size,device=device)

    for epoch in range(nb_epochs):
        t0 = time.time()
        train_loss = 0
        counter=0
        disc_loss_partial=0
        gen_loss_partial=0

        for image_batch_1,image_batch_2, image_batch_3, energy_batch, label_batch in tqdm(dataloader, desc="Training"):
            
            opt_d.zero_grad()

            noise = torch.normal(0, 1, size=(batch_size, latent_size), device=device)
            image_batch_1 = image_batch_1.to(device)
            image_batch_2 = image_batch_2.to(device)
            image_batch_3 = image_batch_3.to(device)
            # energy_breakdown

            sampled_labels = torch.randint(0, nb_classes, size=(batch_size,),device=device)
            sampled_energies = torch.rand( size=(batch_size, 1),device=device)*99+1

            generator_inputs = [noise, sampled_energies]

            """ if nb_classes > 1:
                    # in the case of the ACGAN, we need to append the requested
                    # class to the pre-image of the generator
                    generator_inputs.append(sampled_labels) """

            generated_images = generator(generator_inputs)

            #disc_outputs_real = [torch.ones(batch_size), energy_batch]
            #disc_outputs_fake = [torch.zeros(batch_size), sampled_energies]

            #loss_weights = torch.Tensor([1, 0.05]
            #print(loss_weights.requires_grad)

            
            
            out = discriminator([image_batch_1,image_batch_2, image_batch_3], energy_batch)
            loss_real =   bce_loss(out[0].view(-1), ones) + 0.05 * mae_loss(out[1].view(-1), energy_batch)
            
            loss_real.backward()

            out = discriminator(generated_images, sampled_energies)
            loss_fake =  bce_loss(out[0].view(-1), zeros) + 0.05*mae_loss(out[1].view(-1), sampled_energies)
            
            loss_fake.backward()
            opt_d.step()

            disc_loss_partial+=((loss_real.item() + loss_fake.item()) / 2)
            
            opt_g.zero_grad()


            noise = torch.normal(0, 1, size=(batch_size, latent_size), device=device)
            sampled_energies = torch.rand( size=(batch_size, 1),device=device)*99+1
            combined_inputs = [noise, sampled_energies]
            out=discriminator(generator(combined_inputs),sampled_energies)

            loss_gen=bce_loss(out[0].view(-1), ones) + 0.05*mae_loss(out[1].view(-1), sampled_energies)
            loss_gen.backward()

            opt_g.step()
            gen_loss_partial+=loss_gen.item()


        disc_loss.append(disc_loss_partial/batch_size)
        gen_loss.append(gen_loss_partial/batch_size)
        print('Epoch {:3d} Generator loss: {}'.format(epoch + 1, gen_loss_partial/batch_size))
        print(('Epoch {:3d} Discriminator loss: {}'.format(epoch + 1, disc_loss_partial/batch_size)))
            
        


            


    