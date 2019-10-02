"""
Module containing the decoders.
"""
import numpy as np

import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self, img_size, cs,
                 latent_dim=10):
        r"""Decoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(Decoder, self).__init__()

        # Layer parameters
        self.cs = cs
        kernel_size = 4
        self.latent_dim = latent_dim
        ladder0_dim = 2
        ladder1_dim = 2
        ladder2_dim = 2
        # Shape required to start transpose convs
        self.data_dims = [32, 32, 1]
        self.fs = [self.data_dims[0], self.data_dims[0] // 2, self.data_dims[0] // 4, self.data_dims[0] // 8,
                   self.data_dims[0] // 16]
        cnn_kwargs = dict(stride=2, padding=1)

        # generative 2
        self.affd2_1 = nn.Linear(ladder2_dim, cs[3])
        self.affd2_2 = nn.Linear(cs[3], cs[3])
        self.affd2_3 = nn.Linear(cs[3], cs[3])

        # generative1
        self.affl1_3 = nn.Linear(ladder1_dim, cs[3])
        self.affd1_1 = nn.Linear(cs[3] + cs[3], cs[3])
        self.affd1_2 = nn.Linear(cs[3], cs[3])
        self.affd1_3 = nn.Linear(cs[3], cs[3])

        #generative0
        self.affl0_3 = nn.Linear(ladder0_dim, cs[3])
        self.affd0_1 = nn.Linear(cs[3] + cs[3], int(self.fs[2]*self.fs[2]*self.cs[2]))
        self.convd0_2 = nn.ConvTranspose2d(cs[2], cs[1], kernel_size, **cnn_kwargs)
        self.convd0_3 = nn.ConvTranspose2d(cs[1], self.data_dims[-1], kernel_size, **cnn_kwargs)

    def combine_noise(self, latent, ladder, method='concat'):
        if method is 'concat':
            return torch.cat((latent, ladder), -1)
        else:
            return latent + ladder

    def forward(self, h0_sample, h1_sample, h2_sample):
        # generative2
        h2_gen = torch.relu(self.affd2_1(h2_sample))
        h2_gen = torch.relu(self.affd2_2(h2_gen))
        h2_gen = self.affd2_3(h2_gen)

        # generative1
        h1_sample = torch.relu(self.affl1_3(h1_sample))  # cs[3]
        h1_gen = self.combine_noise(h2_gen, h1_sample)  # cs[3]
        h1_gen = torch.relu(self.affd1_1(h1_gen))
        h1_gen = torch.relu(self.affd1_2(h1_gen))
        h1_gen = self.affd1_3(h1_gen)

        # generative0
        h0_sample = torch.relu(self.affl0_3(h0_sample))
        h0_gen = self.combine_noise(h1_gen, h0_sample)
        h0_gen = self.affd0_1(h0_gen)
        h0_gen = h0_gen.reshape(h0_gen.shape[0], self.cs[2], self.fs[2], self.fs[2])
        h0_gen = self.convd0_2(h0_gen)
        h0_gen = self.convd0_3(h0_gen)

        return h0_gen