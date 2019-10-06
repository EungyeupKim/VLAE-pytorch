import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self, cs,
                 latent_dim=(2, 2, 2)):

        super(Decoder, self).__init__()

        # Layer parameters
        self.cs = cs
        kernel_size = 4
        self.latent_dim0 = latent_dim[0]
        self.latent_dim1 = latent_dim[1]
        self.latent_dim2 = latent_dim[2]
        # Shape required to start transpose convs
        self.data_dims = [32, 32, 1]
        self.fs = [self.data_dims[0], self.data_dims[0] // 2, self.data_dims[0] // 4, self.data_dims[0] // 8,
                   self.data_dims[0] // 16]
        cnn_kwargs = dict(stride=2, padding=1)

        # generative 2
        self.affd2_1 = nn.Linear(self.latent_dim2, cs[3])
        self.bnd2_1 = nn.BatchNorm1d(cs[3], 0.001)
        self.affd2_2 = nn.Linear(cs[3], cs[3])
        self.bnd2_2 = nn.BatchNorm1d(cs[3], 0.001)
        self.affd2_3 = nn.Linear(cs[3], cs[3])

        # generative1
        self.affl1_3 = nn.Linear(self.latent_dim1, cs[3])
        self.bnl1_3 = nn.BatchNorm1d(cs[3], 0.001)
        self.affd1_1 = nn.Linear(cs[3] + cs[3], cs[3])
        self.bnd1_1 = nn.BatchNorm1d(cs[3], 0.001)
        self.affd1_2 = nn.Linear(cs[3], cs[3])
        self.bnd1_2 = nn.BatchNorm1d(cs[3], 0.001)
        self.affd1_3 = nn.Linear(cs[3], cs[3])

        #generative0
        self.affl0_3 = nn.Linear(self.latent_dim0, cs[3])
        self.bnl0_3 = nn.BatchNorm1d(cs[3], 0.001)
        self.affd0_1 = nn.Linear(cs[3] + cs[3], int(self.fs[2]*self.fs[2]*self.cs[2]))
        self.bnd0_1 = nn.BatchNorm1d(int(self.fs[2]*self.fs[2]*self.cs[2]), 0.001)
        self.convd0_2 = nn.ConvTranspose2d(cs[2], cs[1], kernel_size, **cnn_kwargs)
        self.bnd0_2 = nn.BatchNorm2d(cs[1], 0.001)
        self.convd0_3 = nn.ConvTranspose2d(cs[1], self.data_dims[-1], kernel_size, **cnn_kwargs)

    def combine_noise(self, latent, ladder, method='concat'):
        if method is 'concat':
            return torch.cat((latent, ladder), -1)
        else:
            return latent + ladder

    def forward(self, h0_sample, h1_sample, h2_sample):
        # generative2
        h2_gen = self.affd2_1(h2_sample)
        h2_gen = torch.relu(self.bnd2_1(h2_gen))
        h2_gen = self.affd2_2(h2_gen)
        h2_gen = torch.relu(self.bnd2_2(h2_gen))
        h2_gen = self.affd2_3(h2_gen)

        # generative1
        h1_sample = self.affl1_3(h1_sample)
        h1_sample = torch.relu(self.bnl1_3(h1_sample))
        h1_gen = self.combine_noise(h2_gen, h1_sample)
        h1_gen = self.affd1_1(h1_gen)
        h1_gen = torch.relu(self.bnd1_1(h1_gen))
        h1_gen = self.affd1_2(h1_gen)
        h1_gen = torch.relu(self.bnd1_2(h1_gen))
        h1_gen = self.affd1_3(h1_gen)

        # generative0
        h0_sample = self.affl0_3(h0_sample)
        h0_sample = torch.relu(self.bnl0_3(h0_sample))
        h0_gen = self.combine_noise(h1_gen, h0_sample)
        h0_gen = self.affd0_1(h0_gen)
        h0_gen = torch.relu(self.bnd0_1(h0_gen))
        h0_gen = h0_gen.reshape(h0_gen.shape[0], self.cs[2], self.fs[2], self.fs[2])
        h0_gen = self.convd0_2(h0_gen)
        h0_gen = torch.relu(self.bnd0_2(h0_gen))
        h0_gen = torch.sigmoid(self.convd0_3(h0_gen))

        return h0_gen