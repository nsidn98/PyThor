"""
    Variational Autoencoder (Linear)
    Code taken from "@PyTorchLightning/pytorch-lightning-bolts"
"""
import torch
from torch import nn
from torch.nn import functional as F


class Encoder(torch.nn.Module):
    """
        Takes as input a vector, uses a MLP to extract features which
        get split into a mu and sigma vector
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.l1 = DenseBlock(input_dim,hidden_dim)

        self.mu_fc = nn.Linear(hidden_dim,latent_dim)
        self.sigma_fc = nn.Linear(hidden_dim,latent_dim)

    def forward(self, x):
        # NOTE comment the line below, just for testing MNIST
        x = x.view(-1, self.input_dim)
        x = self.l1(x)

        mu = self.mu_fc(x)
        sigma = self.sigma_fc(x)

        return mu, sigma

class Decoder(torch.nn.Module):
    """
        Takes in latent vars and reconstructs an image
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.l1 = DenseBlock(latent_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim,input_dim)

    def forward(self, z):
        x = self.l1(z)
        out = F.sigmoid(self.out(x))
        return out 


class DenseBlock(nn.Module):
    def __init__(self, in_dim, out_dim, drop_p=0.2):
        super().__init__()
        self.drop_p = drop_p
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc_bn = nn.BatchNorm1d(out_dim)
        self.in_dim = in_dim

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc_bn(x)
        x = F.relu(x)
        x = F.dropout(x, self.drop_p)
        return x