import numpy as np
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, img_size, cs,
                 latent_dim=(2, 2, 2)):

        super(Encoder, self).__init__()

        # Layer parameters
        cnn_kwargs = dict(stride=2, padding=1)
        self.cs = cs
        self.latent_dim0 = latent_dim[0]
        self.latent_dim1 = latent_dim[1]
        self.latent_dim2 = latent_dim[2]
        kernel_size = 4
        self.img_size = img_size
        n_chan = self.img_size[0]

        # ladder0
        self.convl0_1 = nn.Conv2d(n_chan, cs[1], kernel_size, **cnn_kwargs)
        self.bnl0_1 = nn.BatchNorm2d(cs[1], 0.001)
        self.convl0_2 = nn.Conv2d(cs[1], cs[2], kernel_size, **cnn_kwargs)
        self.bnl0_2 = nn.BatchNorm2d(cs[2], 0.001)
        self.affl0_mean = nn.Linear(np.product((cs[2], 2 * kernel_size, 2 * kernel_size)), self.latent_dim0)
        self.affl0_stddev = nn.Linear(np.product((cs[2], 2 * kernel_size, 2 * kernel_size)), self.latent_dim0)

        #inference0
        self.convi0_1 = nn.Conv2d(n_chan, cs[1], kernel_size, **cnn_kwargs)
        self.bni0_1 = nn.BatchNorm2d(cs[1], 0.001)
        self.convi0_2 = nn.Conv2d(cs[1], cs[2], kernel_size, **cnn_kwargs)
        self.bni0_2 = nn.BatchNorm2d(cs[2], 0.001)
        self.affi0_3 = nn.Linear(np.product((cs[2], 2 * kernel_size, 2 * kernel_size)), cs[3])

        # ladder1
        self.affl1_1 = nn.Linear(cs[3], cs[3])
        self.bnl1_1 = nn.BatchNorm1d(cs[3], 0.001)
        self.affl1_2 = nn.Linear(cs[3], cs[3])
        self.bnl1_2 = nn.BatchNorm1d(cs[3], 0.001)
        self.affl1_mean = nn.Linear(cs[3], self.latent_dim1)
        self.affl1_stddev = nn.Linear(cs[3], self.latent_dim1)

        # inference1
        self.affi1_1 = nn.Linear(cs[3], cs[3])
        self.bni1_1 = nn.BatchNorm1d(cs[3], 0.001)
        self.affi1_2 = nn.Linear(cs[3], cs[3])
        self.bni1_2 = nn.BatchNorm1d(cs[3], 0.001)
        self.affi1_3 = nn.Linear(cs[3], cs[3])

        # ladder2
        self.affl2_1 = nn.Linear(cs[3], cs[3])
        self.bnl2_1 = nn.BatchNorm1d(cs[3], 0.001)
        self.affl2_2 = nn.Linear(cs[3], cs[3])
        self.bnl2_2 = nn.BatchNorm1d(cs[3], 0.001)
        self.affl2_mean = nn.Linear(cs[3], self.latent_dim2)
        self.affl2_stddev = nn.Linear(cs[3], self.latent_dim2)

    def reparameterize(self, mean, stddev):

        if self.training:
            eps = torch.randn_like(stddev)
            return mean + stddev * eps
        else:
            return mean

    def forward(self, x):

        # ladder0
        h0_ladd = self.convl0_1(x)
        h0_ladd = torch.relu(self.bnl0_1(h0_ladd))
        h0_ladd = self.convl0_2(h0_ladd)
        h0_ladd = torch.relu(self.bnl0_2(h0_ladd))
        h0_ladd = h0_ladd.reshape(h0_ladd.shape[0], -1)
        h0_ladd_mean = self.affl0_mean(h0_ladd)
        h0_ladd_stddev = torch.sigmoid(self.affl0_stddev(h0_ladd)) + 0.001
        h0_sample = self.reparameterize(h0_ladd_mean, h0_ladd_stddev)

        # inference0
        h1_inf = self.convi0_1(x)
        h1_inf = torch.relu(self.bni0_1(h1_inf))
        h1_inf = self.convi0_2(h1_inf)
        h1_inf = torch.relu(self.bni0_2(h1_inf))
        h1_inf = h1_inf.reshape(h1_inf.shape[0], -1)
        h1 = self.affi0_3(h1_inf)

        # ladder1
        h1_ladd = self.affl1_1(h1)
        h1_ladd = torch.relu(self.bnl1_1(h1_ladd))
        h1_ladd = self.affl1_2(h1_ladd)
        h1_ladd = torch.relu(self.bnl1_2(h1_ladd))
        h1_ladd_mean = self.affl1_mean(h1_ladd)
        h1_ladd_stddev = torch.sigmoid(self.affl1_stddev(h1_ladd)) + 0.001
        h1_sample = self.reparameterize(h1_ladd_mean, h1_ladd_stddev)

        # inference1
        h2_inf = self.affi1_1(h1)
        h2_inf = torch.relu(self.bni1_1(h2_inf))
        h2_inf = self.affi1_2(h2_inf)
        h2_inf = torch.relu(self.bni1_2(h2_inf))
        h2 = self.affi1_1(h2_inf)

        # ladder2
        h2_ladd = self.affl2_1(h2)
        h2_ladd = torch.relu(self.bnl2_1(h2_ladd))
        h2_ladd = self.affl2_2(h2_ladd)
        h2_ladd = torch.relu(self.bnl2_2(h2_ladd))
        h2_ladd_mean = self.affl2_mean(h2_ladd)
        h2_ladd_stddev = torch.sigmoid(self.affl2_stddev(h2_ladd)) + 0.001
        h2_sample = self.reparameterize(h2_ladd_mean, h2_ladd_stddev)

        return h0_ladd_mean, h0_ladd_stddev, h0_sample, h1_ladd_mean, h1_ladd_stddev, h1_sample, h2_ladd_mean, h2_ladd_stddev, h2_sample
