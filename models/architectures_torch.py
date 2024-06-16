import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import math

from ops_pytorch import (minibatch_discriminator, LocallyConnected2d,minibatch_output_shape,
                 Dense3D, sparsity_level)

def sparse_softmax(x):
    x = F.relu(x)
    e = torch.exp(x - torch.max(x, dim=(1, 2, 3), keepdim=True).values)
    s = torch.sum(e, dim=(1, 2, 3), keepdim=True)
    return e / s


class build_Generator(nn.Module):
    def __init__(self, latent_dim, nb_rows, nb_cols):
        super(build_Generator, self).__init__()
        self.fc = nn.Linear(latent_dim, (nb_rows + 2) * (nb_cols + 2) * 36)
        self.reshape_size = (nb_rows + 2, nb_cols + 2, 36)

        self.conv1 = nn.Conv2d(36, 16, kernel_size=2,padding='same')
        self.lrelu = nn.LeakyReLU()
        self.batch_norm = nn.BatchNorm2d(16)

        # Using custom LocallyConnected2d layers
        self.local_conv1 = LocallyConnected2d(16, 6, [nb_rows+2,nb_cols+2], kernel_size=2, bias=0)
        self.local_conv2 = LocallyConnected2d(6, 1, [nb_rows+1, nb_cols+1], kernel_size=2, bias=0)

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
        return x
    
class build_Discriminator(nn.Module):
    def __init__(self, mbd=False, sparsity=False, sparsity_mbd=False,sizes=[10,10]):
        super(build_Discriminator, self).__init__()
        self.mbd = mbd
        self.sparsity = sparsity
        self.sparsity_mbd = sparsity_mbd
        self.sizes = sizes

        self.conv1 = nn.Conv2d(1, 64, kernel_size=2, padding='same')
        self.lrelu = nn.LeakyReLU()

        self.zero_pad = nn.ZeroPad2d(1)
        self.local_conv1 = LocallyConnected2d(64, 16, input_size=(sizes[0]+2, sizes[1]+2), kernel_size=3, stride=(1,2), bias=0)
        self.batch_norm1 = nn.BatchNorm2d(16)

        self.local_conv2 = LocallyConnected2d(16, 8, input_size=(sizes[0]+2, math.floor((sizes[1]-1)/2.) +3), kernel_size=2, bias=0)
        self.batch_norm2 = nn.BatchNorm2d(8)

        self.local_conv3 = LocallyConnected2d(8, 8, input_size=(sizes[0]+3,  math.floor((sizes[1]-1)/2.)+4), kernel_size=2, stride=(1,2), bias=0)
        self.batch_norm3 = nn.BatchNorm2d(8)
        self.flatten = nn.Flatten()
        self.outShape = (sizes[0]+2)*(math.floor((math.floor((sizes[1]-1)/2.)+2)/2)+1)*8
     

        if self.mbd or self.sparsity or self.sparsity_mbd:
            self.minibatch_featurizer = minibatch_discriminator
            nb_features = 10
            vspace_dim = 10
            self.dense3d_1 = Dense3D(first_dim=nb_features, last_dim=vspace_dim, input_shape=(16,self.outShape) )
            self.dense3d_2 = Dense3D(first_dim=nb_features, last_dim=vspace_dim, input_shape=(16,1))
            self.activation_tanh = nn.Tanh()
            self.sparsity_detector = sparsity_level 

    def forward(self, image):
        #x = image.permute(0, 3, 1, 2)
        x=image
        x = self.conv1(x)
        x = self.lrelu(x)

        x = self.zero_pad(x) 

        x = self.local_conv1(x)
        x = self.lrelu(x)
        x = self.batch_norm1(x)

        x = self.zero_pad(x)
        
        x = self.local_conv2(x)
        x = self.lrelu(x)
        x = self.batch_norm2(x)

        x = self.zero_pad(x)
    
        x = self.local_conv3(x)
        x = self.lrelu(x)
        x = self.batch_norm3(x)

        if self.mbd or self.sparsity or self.sparsity_mbd:
            features = [x]
            if self.mbd:
                K_x = self.dense3d_1(x)
                features.append(self.activation_tanh(self.minibatch_featurizer(K_x)))


            if self.sparsity or self.sparsity_mbd:
                empirical_sparsity = self.sparsity_detector(image)
                if self.sparsity:
                    features.append(empirical_sparsity)
                if self.sparsity_mbd:
                    K_sparsity = self.dense3d_2(empirical_sparsity)
                    features.append(self.activation_tanh(self.minibatch_featurizer(K_sparsity)))

            return torch.cat(features, dim=1)
        else:
            return x
