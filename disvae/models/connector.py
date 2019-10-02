"""
Module containing the encoders.
"""
import numpy as np

import torch
from torch import nn

class Connector(nn.Module):
    def __init__(self, cs, latent_dim=10):
        r"""Encoder of the model proposed in [1].

        Parameters
        ----------
        latent_inputdim : int
        latent_outputdim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 affine layers (each with z size = 2 x previous z size)
        - Latent distribution:
            - 4 fully connected layers  (log variance and mean for 10 Gaussians)

        """
        super(Connector, self).__init__()

        # Layer parameters
        self.cs = cs
        self.latent_dim = latent_dim # 10
        self.hdim1 = self.latent_dim # 10
        self.hdim2 = self.latent_dim # 10
        self.hdim3 = self.latent_dim # 10
        self.latent_outdim = self.latent_dim # 10

        # connecting layers of VLAE
        hid_channel = 1024
        ladder1_dim = 2
        ladder2_dim = 2

        # ladder1
        self.affl1_1 = nn.Linear(cs[3], cs[3])  # 20, 20
        self.affl1_2 = nn.Linear(cs[3], cs[3])  # 20, 20
        self.affl1_3 = nn.Linear(ladder1_dim, cs[3])
        self.affl1_mean = nn.Linear(cs[3], ladder1_dim)  # 20, 10
        self.affl1_stddev = nn.Linear(cs[3], ladder1_dim)  # 20, 10

        # inference1
        self.affi1_1 = nn.Linear(cs[3], cs[3])  # 20, 20
        self.affi1_2 = nn.Linear(cs[3], cs[3])  # 20, 20
        self.affi1_3 = nn.Linear(cs[3], cs[3])  # 20, 20

        # ladder2
        self.affl2_1 = nn.Linear(cs[3], cs[3])  # 20, 20
        self.affl2_2 = nn.Linear(cs[3], cs[3])  # 20, 20
        self.affl2_mean = nn.Linear(cs[3], ladder2_dim)  # 20, 10
        self.affl2_stddev = nn.Linear(cs[3], ladder2_dim)  # 20, 10

        # generative1
        self.affd1_1 = nn.Linear(cs[3] + cs[3], cs[3])
        self.affd1_2 = nn.Linear(cs[3], cs[3])
        self.affd1_3 = nn.Linear(cs[3], cs[3])

        # generative 2
        self.affd2_1 = nn.Linear(ladder2_dim, cs[3])
        self.affd2_2 = nn.Linear(cs[3], cs[3])
        self.affd2_3 = nn.Linear(cs[3], cs[3])

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)


        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def combine_noise(self, latent, ladder, method='add'):
        if method is 'concat':
            return torch.cat((latent, ladder), -1)
        else:
            return latent + ladder

    def forward(self, h1, h0_sample):

        #ladder1
        h1_ladd = torch.relu(self.affl1_1(h1))
        h1_ladd = torch.relu(self.affl1_2(h1_ladd))
        h1_ladd_mean = self.affl1_mean(h1_ladd)
        h1_ladd_stddev = torch.sigmoid(self.affl1_stddev(h1_ladd))
        h1_sample = self.reparameterize(h1_ladd_mean, h1_ladd_stddev)

        #inference1
        h2_inf = torch.relu(self.affi1_1(h1))
        h2_inf = torch.relu(self.affi1_2(h2_inf))
        h2 = self.affi1_1(h2_inf)

        #ladder2
        h2_ladd = torch.relu(self.affl2_1(h2))
        h2_ladd = torch.relu(self.affl2_2(h2_ladd))
        h2_ladd_mean = self.affl1_mean(h2_ladd)
        h2_ladd_stddev = torch.sigmoid(self.affl1_stddev(h2_ladd))
        h2_sample = self.reparameterize(h2_ladd_mean, h2_ladd_stddev)

        #generative2
        h2_gen = torch.relu(self.affd2_1(h2_sample))
        h2_gen = torch.relu(self.affd2_2(h2_gen))
        h2_gen = self.affd2_3(h2_gen)

        #generative1
        h1_sample = torch.relu(self.affl1_3(h1_sample)) # cs[3]
        h1_gen = self.combine_noise(h2_gen, h1_sample) # cs[3]
        h1_gen = torch.relu(self.affd1_1(h1_gen))
        h1_gen = torch.relu(self.affd1_2(h1_gen))
        h1_gen = self.affd1_3(h1_gen)


        return h1_gen, h1_ladd_mean, h1_ladd_stddev, h2_ladd_mean, h2_ladd_stddev