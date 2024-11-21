import torch
import torch.nn as nn

from ..utils import AffineForPSA, ModifiedBatchNorm1d
from .psa import PersonalSelfAttention
# from .dbg_psa import PersonalSelfAttention
from .lerp_psa import LerpPersonalSelfAttention
# from .other_psa import FastPersonalSelfAttention as LerpPersonalSelfAttention


class ResPSA(nn.Module):
    def __init__(self, input_features, expansion_features,
            add_res=True, nrm_input=False,
            psa_type='psa', **kwargs):
        super().__init__()

        self.input_features = input_features
        self.expansion_features = expansion_features
        self.add_res = add_res

        self.psa_ln = None
        if nrm_input:
            self.psa_ln = ModifiedBatchNorm1d(input_features)

        assert psa_type in ['psa', 'lerp_psa']
        if psa_type == 'psa':
            self.psa = PersonalSelfAttention(input_features, expansion_features, **kwargs)
        elif psa_type == 'lerp_psa':
            self.psa = LerpPersonalSelfAttention(input_features, expansion_features, **kwargs)

    def forward(self, x):
        if self.psa_ln is not None:
            x = self.psa_ln(x)

        # copy residual after any layernorm
        x_res = x

        x = self.psa(x)

        if self.add_res:
            x = x + x_res

        return x

    def batchnorm_to_affine(self):
        if type(self.psa_ln) == ModifiedBatchNorm1d:
            with torch.no_grad():
                bn = self.psa_ln.bn
                mean = bn.running_mean
                var_sqrt = torch.sqrt(bn.running_var + bn.eps)
                beta = bn.weight
                gamma = bn.bias

                w_s = beta / var_sqrt
                b_o = -mean / var_sqrt * beta + gamma

            self.psa_ln = AffineForPSA(w_s.shape[0]).to(w_s.device)
            with torch.no_grad():
                self.psa_ln.weight.set_(w_s)
                self.psa_ln.bias.set_(b_o)
