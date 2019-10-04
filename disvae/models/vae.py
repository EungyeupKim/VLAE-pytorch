"""
Module containing the main VAE class.
"""
import torch
from torch import nn, optim
from torch.nn import functional as F
from disvae.models.encoders import Encoder
from disvae.models.decoders import Decoder

class VLAE(nn.Module):
    def __init__(self, latent_dim, cs):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super(VLAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = (1, 32, 32)
        self.encoder = Encoder(self.img_size, cs, self.latent_dim)
        self.decoder = Decoder(cs, self.latent_dim)

    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        h0_ladd_mean, h0_ladd_stddev, h0_sample, h1_ladd_mean, h1_ladd_stddev, h1_sample, h2_ladd_mean, h2_ladd_stddev, h2_sample = self.encoder(x)
        gen_img = self.decoder(h0_sample, h1_sample, h2_sample)

        return gen_img, h0_ladd_mean, h0_ladd_stddev, h1_ladd_mean, h1_ladd_stddev, h2_ladd_mean, h2_ladd_stddev