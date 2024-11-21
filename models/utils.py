import torch
from torch import nn


class ModifiedBatchNorm1d(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        self.bn = nn.BatchNorm1d(in_features)


    def forward(self, x):
        og_shape = x.shape
        x = x.reshape(og_shape[0], -1, self.in_features)
        x = x.transpose(-1, -2)
        x = self.bn(x)
        x = x.transpose(-1, -2)
        x = x.reshape(og_shape)
        return x


class ResidualScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.ones(1))

    def forward(self, x):
        return x * self.weight


class Mul(nn.Module):
    def __init__(self, dim, alpha_init=1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))

        with torch.no_grad():
            self.weight.fill_(alpha_init)

    def forward(self, x):
        return x * self.weight


class Affine(nn.Module):
    def __init__(self, dim, alpha_init=1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

        with torch.no_grad():
            self.weight.fill_(alpha_init)

    def forward(self, x):
        return x * self.weight + self.bias


# Create a copy class for Affine so its treated differently outside
# with loops that type-check for 'affine' class objects
class AffineForPSA(Affine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AvgPool(nn.Module):
    def __init__(self, dim, n):
        super().__init__()
        self.dim = dim
        self.n = n


    def forward(self, x):
        return x.sum(dim=self.dim) / self.n
