
import torch
from torch import nn, optim
from torch.nn import functional as F
from disvae.models.encoders import Encoder
from disvae.models.decoders import Decoder

class VLAE(nn.Module):
    def __init__(self, args, latent_dim, cs):
        super(VLAE, self).__init__()

        self.args = args
        self.latent_dim = latent_dim
        self.img_size = (1, 32, 32)
        self.encoder = Encoder(self.args, self.img_size, cs, self.latent_dim)
        self.decoder = Decoder(self.args, cs, self.latent_dim)

    def forward(self, x):

        h0_ladd_mean, h0_ladd_stddev, h0_sample, h1_ladd_mean, h1_ladd_stddev, h1_sample, h2_ladd_mean, h2_ladd_stddev, h2_sample = self.encoder(x)
        gen_img = self.decoder(h0_sample, h1_sample, h2_sample)
        return gen_img, h0_ladd_mean, h0_ladd_stddev, h1_ladd_mean, h1_ladd_stddev, h2_ladd_mean, h2_ladd_stddev