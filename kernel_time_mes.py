import sys
import numpy as np

import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark

from models.ds_modules import dsnn

from models.psa_modules.psa import PersonalSelfAttention as OG_PersonalSelfAttention
# from models.psa import PersonalSelfAttention
# from models.psa_modules.other_psa import FastPersonalSelfAttention as LerpPersonalSelfAttention
# from models.psa_modules.other_psa_v2 import FastPersonalSelfAttention as LerpPersonalSelfAttention
from models.psa_modules.lerp_psa import LerpPersonalSelfAttention
from models.psa_modules.respsa import ResPSA
from models.utils import Affine


EVAL_TEST=False

n1 = 10
lmbda = 1e-4 # regularization weight
lipschitz = False # lipschitz control
def run_model(model, x, optimizer, aux_optimizer):
    if not EVAL_TEST:
        model.train()
        for i in range(n1):
            optimizer.zero_grad()
            if aux_optimizer is not None:
                aux_optimizer.zero_grad()

            z = model(x)
            loss = z.sum()
            if aux_optimizer is not None:
                if lipschitz is True:
                    loss = loss + lmbda * model.BV2()
                else:
                    loss = loss + lmbda * model.TV2()

            loss.backward()

            optimizer.step()
            if aux_optimizer is not None:
                aux_optimizer.step()

    if EVAL_TEST:
        model.eval()
        for i in range(n1):
            with torch.no_grad():
                model(x)


def run_test(act):
    assert act in ['none', 'bspline', 'relu', 'gelu', 'unrolled_psa', 'lerp_psa']

    nblocks = 4

    isize = 768
    ssize = 512
    bsize = 16

    grid_size = 2**6
    exsize = 8
    save_memory = True

    save_grid_outputs = True
    epsilon = 1.0

    aux_optimizer = None

    if act == 'none':
        model = nn.Sequential(
            *([nn.Linear(isize, isize)]*nblocks)
        ).to('cuda')

    if act == 'relu':
        model = nn.Sequential(
            *([nn.Linear(isize, isize),
            nn.ReLU()]*nblocks)
        ).to('cuda')

    if act == 'gelu':
        model = nn.Sequential(
            *([nn.Linear(isize, isize),
            nn.GELU()]*nblocks)
        ).to('cuda')

    if act == 'bspline':
        opt_params = {
            'size': grid_size + 1,
            'range_': 4,
            'init': 'leaky_relu',
            'save_memory': save_memory
        }

        class Model(dsnn.DSModule):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    *([dsnn.DeepBSpline('fc', isize, **opt_params),
                    nn.Linear(isize, isize)]*nblocks)
                )

            def forward(self, x):
                return self.layers(x)

        model = Model().to('cuda')

        optimizer = torch.optim.AdamW(model.parameters_no_deepspline(), lr=1e-3, weight_decay=5e-4)
        aux_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

        # model = nn.Sequential(
        #     nn.Linear(isize, isize),
        #     dsnn.DeepBSpline('fc', isize, **opt_params)
        # ).to('cuda')
        # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

    if act == 'unrolled_psa':
        model = nn.Sequential(
            *([OG_PersonalSelfAttention(isize, exsize, epsilon=epsilon, save_memory=save_memory, grid_size=grid_size),
            nn.Linear(isize, isize)]*nblocks)

        ).to('cuda')

    if act == 'lerp_psa':
        psa_kwargs = {
            'add_res': False,
            'affine_output': True,
            'alpha_init': 1.0,
            'epsilon': epsilon,
            'distill_init': True,
            'lerp_psa': True,
            'grid_size': grid_size,
            'grid_range': 4,
            'save_memory': save_memory,
            'save_grid_outputs': save_grid_outputs,
            'nrm_input': False
        }
        model = nn.Sequential(
            *([LerpPersonalSelfAttention(isize, exsize, **psa_kwargs),
            nn.Linear(isize, isize),] * nblocks)

        ).to('cuda')

    print(model)

    x = torch.randn(bsize * ssize, isize).to('cuda')

    # B-Spline uses split optimizers
    if act != 'bspline':
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

    num_threads = torch.get_num_threads()

    rts = []
    for i in range(10):
        t1 = benchmark.Timer(
            stmt='run_model(model, x, optimizer, aux_optimizer)',
            setup='from __main__ import run_model',
            globals={'model': model, 'x': x, 'optimizer': optimizer, 'aux_optimizer': aux_optimizer},
            num_threads=num_threads)
        mes = t1.timeit(1)
        rts.append(mes.mean)

    print('\n' + '-' * 75 + '\n')
    rts = np.array(rts) * 1e3
    print(f"Mean +- STD : {np.mean(rts):.2f} + {np.std(rts):.2f} ms")
    print(f"Median : {np.median(rts):.2f} ms")

    max_mem_mb = torch.cuda.max_memory_allocated() / (1024.**2)
    print(f"Max memory consumed (MB): {max_mem_mb:.2f}")


if __name__ == '__main__':
    assert len(sys.argv) > 1, "Please provide an activation to test"
    run_test(sys.argv[1])
