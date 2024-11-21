import random
import argparse
from tqdm import tqdm
from pprint import pprint
import math

import numpy as np
import scipy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from models.utils import ModifiedBatchNorm1d

from models.ds_modules import dsnn
from models.psa_modules.psa import PersonalSelfAttention
from models.psa_modules.lerp_psa import LerpPersonalSelfAttention


parser = argparse.ArgumentParser()
parser.add_argument('--teacher-nl-act', type=str, choices=['psa','bspline', 'base'], default='base')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--disable-tqdm', action='store_true', default=False)
parser.add_argument('--student-nl-act', type=str, choices=['psa','bspline', 'base'], default='psa')
parser.add_argument('--bspline-lambda', default=1e-3, type=float)
parser.add_argument('--save', default=None, type=str)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--nondeter', action='store_true', default=False)
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if not args.nondeter:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

def Feynman(x, eqn=1):
    assert x.shape[-1] == 2
    a = x[..., 0]
    b = x[..., 1]
    if eqn == 1:
        return 1 + a * torch.sin(b)
    if eqn == 2:
        # return 1 / (1 + (a * b))
        return (a - 1) * b
    if eqn == 3:
        return b * torch.exp(-a)
    if eqn == 4:
        return torch.cos(a) + b*(torch.cos(a)**2)
    if eqn == 5:
        return torch.sqrt(1 + a**2 + b**2)

    assert x.shape[-1] == 3
    c = x[..., 2]
    if eqn == 6:
        return a + b * c
    if eqn == 7:
        return c * torch.sqrt(a**2 + b**2)
    if eqn == 8:
        return a * (1 + b * np.cos(c))

    raise NotImplementedError

for i_fn in range(1, 9):

    def test_function(x):
        return Feynman(x, i_fn)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    input_size = 2 if i_fn <= 5 else 3
    output_size = 1
    hidden_size = input_size * 128
    expansion_size = 16
    # grid_size = 16 * 3
    grid_size = 2**8
    batch_size = 1024

    x_min, x_max = -1, 1
    x_scale = x_max - x_min

    sample_x = torch.rand(batch_size, input_size)
    sample_x = (sample_x * x_scale) + x_min

    # Uniform distribution mean std
    sample_x_mu = (x_max + x_min) / 2
    sample_x_std = math.sqrt(((x_max - x_min) ** 2) / 12)

    sample_target = test_function(sample_x)
    sample_target_mu = sample_target.mean()
    sample_target_std = sample_target.std()
    # print(sample_target_mu.numpy(), sample_target_std.numpy())

    using_bspline = False

    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    )

    # model = nn.Sequential(
    #     PersonalSelfAttention(input_size, expansion_size, epsilon=1e-5,
    #         distill_init=True, corrected_grid_init=True),
    #     nn.Linear(input_size, hidden_size),
    #     PersonalSelfAttention(hidden_size, expansion_size, epsilon=1e-5,
    #         distill_init=True, corrected_grid_init=True),
    #     nn.Linear(hidden_size, output_size)
    # )

    # model = nn.Sequential(
    #     LerpPersonalSelfAttention(input_size, expansion_size, epsilon=1e-5, distill_init=True,
    #         grid_size=grid_size, corrected_grid_init=True),
    #     nn.Linear(input_size, hidden_size),
    #     LerpPersonalSelfAttention(hidden_size, expansion_size, epsilon=1e-5, distill_init=True,
    #         grid_size=grid_size, corrected_grid_init=True),
    #     nn.Linear(hidden_size, output_size)
    # )

    # using_bspline = True
    # opt_params = {
    #     'size': grid_size + 1,
    #     'range_': 4,
    #     'init': 'leaky_relu',
    #     'save_memory': False
    # }
    # class Model(dsnn.DSModule):
    #     def __init__(self):
    #         super().__init__()
    #         self.layers = nn.Sequential(
    #             dsnn.DeepBSpline('fc', input_size, **opt_params),
    #             nn.Linear(input_size, hidden_size),
    #             dsnn.DeepBSpline('fc', hidden_size, **opt_params),
    #             nn.Linear(hidden_size, output_size)
    #         )
    #     def forward(self, x):
    #         return self.layers(x)
    # model = Model()

    lmbda = args.bspline_lambda # regularization weight
    lipschitz = False # lipschitz control

    def plot_fit(model):
        criterion = nn.MSELoss()
        if not using_bspline:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        else:
            optimizer = optim.AdamW(model.parameters_no_deepspline(), lr=1e-3, weight_decay=1e-5)
            aux_optimizer = optim.Adam(model.parameters_deepspline())

        # x_train = torch.from_numpy(np.linspace(x_min, x_max, 256)).float().reshape(-1, 1)
        # x_train = ((torch.rand(batch_size, 1) * x_scale) + x_min)

        model.train()
        # for i in tqdm(range(10000)):
        for i in range(10000):
            x_train = torch.rand(batch_size, input_size)
            x_train = (x_train * x_scale) + x_min
            x_train = (x_train - sample_x_mu) / sample_x_std

            # x_train.requires_grad = True
            target = test_function(x_train)#.reshape(-1, 1)
            target = (target - sample_target_mu) / sample_target_std

            optimizer.zero_grad()
            if using_bspline:
                aux_optimizer.zero_grad()

            output = model(x_train).squeeze()
            loss = criterion(output, target)

            if using_bspline:
                if lipschitz is True:
                    loss = loss + lmbda * model.BV2()
                else:
                    loss = loss + lmbda * model.TV2()

            loss.backward()
            optimizer.step()
            if using_bspline:
                aux_optimizer.step()

        model.eval()

        x_eval = torch.rand(batch_size, input_size)
        x_eval = (x_eval * x_scale) + x_min
        x_eval = (x_eval - sample_x_mu) / sample_x_std

        # x_eval.requires_grad = True
        target_eval = test_function(x_eval)#.reshape(-1, 1)
        target_eval = (target_eval - sample_target_mu) / sample_target_std

        output = model(x_eval).squeeze()
        loss = criterion(output, target_eval)
        print("{:.3E}".format(loss.item()))

        # plt.plot(x, y_fit, color=color, label=label_name, linestyle=linestyle, alpha=alpha)

    plot_fit(model)
