"""
Module containing the encoders.
"""
import numpy as np
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, img_size, cs,
                 latent_dim=10):
        r"""Encoder of the model proposed in [1].

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
        super(Encoder, self).__init__()

        # Layer parameters
        cnn_kwargs = dict(stride=2, padding=1)
        self.cs = cs
        ladder0_dim = 2
        ladder1_dim = 2
        ladder2_dim = 2
        kernel_size = 4
        self.latent_dim = latent_dim
        self.img_size = img_size
        n_chan = self.img_size[0]

        # ladder0
        self.convl0_1 = nn.Conv2d(n_chan, cs[1], kernel_size, **cnn_kwargs)
        self.convl0_2 = nn.Conv2d(cs[1], cs[2], kernel_size, **cnn_kwargs)
        self.affl0_mean = nn.Linear(np.product((cs[2], 2 * kernel_size, 2 * kernel_size)), ladder0_dim)
        self.affl0_stddev = nn.Linear(np.product((cs[2], 2 * kernel_size, 2 * kernel_size)), ladder0_dim)

        #inference0
        self.convi0_1 = nn.Conv2d(n_chan, cs[1], kernel_size, **cnn_kwargs)
        self.convi0_2 = nn.Conv2d(cs[1], cs[2], kernel_size, **cnn_kwargs)
        self.affi0_3 = nn.Linear(np.product((cs[2], 2 * kernel_size, 2 * kernel_size)), cs[3])

        # ladder1
        self.affl1_1 = nn.Linear(cs[3], cs[3])  # 20, 20
        self.affl1_2 = nn.Linear(cs[3], cs[3])  # 20, 20
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

    def forward(self, x):

        # ladder0
        h0_ladd = torch.relu(self.convl0_1(x)) # 64 x 64 x 16 x 16
        h0_ladd = torch.relu(self.convl0_2(h0_ladd)) # 64 x 128 x 8 x 8
        h0_ladd = h0_ladd.reshape(h0_ladd.shape[0], -1)
        h0_ladd_mean = self.affl0_mean(h0_ladd)
        h0_ladd_stddev = torch.sigmoid(self.affl0_stddev(h0_ladd))
        h0_sample = self.reparameterize(h0_ladd_mean, h0_ladd_stddev)

        # inference0
        h1_inf = torch.relu(self.convi0_1(x))
        h1_inf = torch.relu(self.convi0_2(h1_inf))
        h1_inf = h1_inf.reshape(h1_inf.shape[0], -1)
        h1 = self.affi0_3(h1_inf)

        # ladder1
        h1_ladd = torch.relu(self.affl1_1(h1))
        h1_ladd = torch.relu(self.affl1_2(h1_ladd))
        h1_ladd_mean = self.affl1_mean(h1_ladd)
        h1_ladd_stddev = torch.sigmoid(self.affl1_stddev(h1_ladd))
        h1_sample = self.reparameterize(h1_ladd_mean, h1_ladd_stddev)

        # inference1
        h2_inf = torch.relu(self.affi1_1(h1))
        h2_inf = torch.relu(self.affi1_2(h2_inf))
        h2 = self.affi1_1(h2_inf)

        # ladder2
        h2_ladd = torch.relu(self.affl2_1(h2))
        h2_ladd = torch.relu(self.affl2_2(h2_ladd))
        h2_ladd_mean = self.affl1_mean(h2_ladd)
        h2_ladd_stddev = torch.sigmoid(self.affl1_stddev(h2_ladd))
        h2_sample = self.reparameterize(h2_ladd_mean, h2_ladd_stddev)

        return h0_ladd_mean, h0_ladd_stddev, h0_sample, h1_ladd_mean, h1_ladd_stddev, h1_sample, h2_ladd_mean, h2_ladd_stddev, h2_sample
