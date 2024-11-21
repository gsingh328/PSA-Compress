import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from models.ds_modules import dsnn

from models.utils import ModifiedBatchNorm1d
from models.psa_modules.psa import PersonalSelfAttention
from models.psa_modules.lerp_psa import LerpPersonalSelfAttention

plt.rcParams['savefig.dpi'] = 300
plt.rcParams.update({'font.size': 14})


torch.random.manual_seed(2023)
np.random.seed(2023)


def test_function_sym(x, lib='torch'):
    # Assuming x is bound to [-1, 1]
    y_mu, y_std = -0.77, 3.7
    # x = (x * 5) + 5
    if lib == 'torch':
        y = -x * torch.sin(x)
    elif lib == 'numpy':
        y = -x * np.sin(x)
    else:
        raise NotImplementedError

    return (y - y_mu) / y_std


def test_function_assym(x, lib='torch'):
    # Softplus
    beta = 0.4
    if lib == 'torch':
        z = (1 / beta) * torch.log(1 + torch.exp(beta * x))
    elif lib == 'numpy':
        z = (1 / beta) * np.log(1 + np.exp(beta * x))
    else:
        raise NotImplementedError

    y_mu, y_std = 1.7277768, 2.085805
    C_A, C_B, C_C, C_D = 1.0, 3.0, 1.4, 0.3
    if lib == 'torch':
        y = C_A * torch.exp(-x / C_B) + C_C * torch.sin(z / C_D)
    elif lib == 'numpy':
        y = C_A * np.exp(-x / C_B) + C_C * np.sin(z / C_D)
    else:
        raise NotImplementedError

    return (y - y_mu) / y_std
    # return -4*x



# test_function = test_function_sym
# x_min, x_max = 0, 10

test_function = test_function_assym
x_min, x_max = -5, 5

x_scale = x_max - x_min


def plot_fit(model, label_name, color, linestyle, alpha=1.0):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

    batch_size = 1024
    # x_train = torch.from_numpy(np.linspace(x_min, x_max, 256)).float().reshape(-1, 1)
    # x_train = ((torch.rand(batch_size, 1) * x_scale) + x_min)

    model.train()
    for i in range(10000):
        # x_train = (torch.randn(batch_size, 1) / 3).clip(-1, 1)
        x_train = ((torch.rand(batch_size, 1) * x_scale) + x_min)
        # x_train.requires_grad = True
        target = test_function(x_train)

        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    model.eval()
    x = np.linspace(x_min, x_max, 256)
    y_fit = model(
        torch.from_numpy(x).float().reshape(-1, 1)
    ).detach().cpu().numpy()

    plt.plot(x, y_fit, color=color, label=label_name, linestyle=linestyle, alpha=alpha)

x = np.linspace(x_min, x_max, 256)
y = test_function(torch.from_numpy(x).float(), lib='torch').numpy()
plt.plot(x, y, label='Target', color='blue', linestyle='dashed')

print(y.mean(), y.std())

torch.random.manual_seed(2023)
np.random.seed(2023)

psa_model = nn.Sequential(
    ModifiedBatchNorm1d(1),     # This is necessary since training data is random
    LerpPersonalSelfAttention(1, 8, epsilon=1, distill_init=False,
        grid_size=2**3, corrected_grid_init=True),
)
plot_fit(psa_model, 'LERP-PSA-8', 'darkorange', 'dotted', alpha=0.7)

torch.random.manual_seed(2023)
np.random.seed(2023)

psa_model = nn.Sequential(
    ModifiedBatchNorm1d(1),     # This is necessary since training data is random
    LerpPersonalSelfAttention(1, 8, epsilon=1, distill_init=False,
        grid_size=2**4, corrected_grid_init=True),
)
plot_fit(psa_model, 'LERP-PSA-16', 'darkorange', 'solid', alpha=0.7)

plt.legend(fontsize="12", loc ="upper right")
plt.savefig('univariate_plot.png', bbox_inches="tight")


# torch.random.manual_seed(2023)
# np.random.seed(2023)

# psa_model = nn.Sequential(
#     ModifiedBatchNorm1d(1),     # This is necessary since training data is random
#     PersonalSelfAttention(1, 8, epsilon=0.0, distill_init=False),
# )
# # plot_fit(psa_model, 'PSA', 'red', 'solid')
# plot_fit(psa_model, 'PSA-8', 'red', 'solid', alpha=0.7)


# torch.random.manual_seed(2023)
# np.random.seed(2023)

# opt_params = {
#     'size': 17,
#     'range_': 4,
#     'init': 'leaky_relu',
#     'save_memory': False
# }
# psa_model = nn.Sequential(
#     ModifiedBatchNorm1d(1),     # This is necessary since training data is random
#     dsnn.DeepBSpline('fc', 1, **opt_params),
# )
# plot_fit(psa_model, 'B-Spline-17', 'green', 'dotted', alpha=0.7)

# torch.random.manual_seed(2023)
# np.random.seed(2023)

# opt_params = {
#     'size': 33,
#     'range_': 4,
#     'init': 'leaky_relu',
#     'save_memory': False
# }
# psa_model = nn.Sequential(
#     ModifiedBatchNorm1d(1),     # This is necessary since training data is random
#     dsnn.DeepBSpline('fc', 1, **opt_params),
# )
# plot_fit(psa_model, 'B-Spline-33', 'green', 'dashdot', alpha=0.7)

# plt.legend(fontsize="12", loc ="upper right")
# plt.savefig('univariate_plot_assym_2.png', bbox_inches="tight")

