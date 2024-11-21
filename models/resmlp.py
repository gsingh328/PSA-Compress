import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from .utils import Affine, Mul, AvgPool, ModifiedBatchNorm1d
from .quantization_helper import (quant_forward)


_ACT_FN = nn.ReLU
# _ACT_FN = nn.GELU
# _ACT_FN = nn.SiLU


def gen_param_dict(m, quantized):
    rt = {}
    w = m.weight.detach()
    if quantized:
        w, _ = quant_forward(w, m.weight_abs_max, m.weight_bits, dequantize=False)
    rt['w'] = w.cpu().numpy().astype(np.int32)
    assert list(rt['w'].shape) == list(m.weight.shape)

    if hasattr(m, 'bias') and m.bias is not None:
        b = m.bias.detach()
        if quantized:
            b, _ = quant_forward(b, m.bias_abs_max, m.bias_bits, dequantize=False)
        rt['b'] = b.cpu().numpy().astype(np.int32)
        assert list(rt['b'].shape) == list(m.bias.shape)

    if quantized:
        s = m.requant_s.detach()
        rt['s'] = s.cpu().numpy().astype(np.int32)
        assert list(rt['s'].shape) == list(m.requant_s.shape)
    return rt


def equalize_scales(*args):
    scale_max = min(args)
    for m in args:
        with torch.no_grad():
            m.fill_(scale_max)


class CrossPatchSublayer(nn.Module):
    def __init__(self, in_patches, in_channels, layerscale_init=0.1):
        super().__init__()

        self.affine_1 = ModifiedBatchNorm1d(in_channels)
        self.affine_2 = Affine(in_channels, alpha_init=layerscale_init)

        self.layers = nn.Linear(in_patches, in_patches)

        # A helper layer to absorb affines
        self.res_affine = None


    def forward(self, x):
        if not hasattr(self, 'res_affine') or self.res_affine is None:
            x_res = x
        else:
            x_res = self.res_affine(x)

        x = self.affine_1(x).transpose(1,2)
        x = self.layers(x).transpose(1,2)
        x = self.affine_2(x)

        # x = self.affine_1(x.transpose(1,2))
        # x = self.layers(x)
        # x = self.affine_2(x).transpose(1,2)

        x = x + x_res
        return x


    def batchnorm_to_affine(self):
        if type(self.affine_1) != Affine:
            with torch.no_grad():
                bn = self.affine_1.bn
                mean = bn.running_mean
                var_sqrt = torch.sqrt(bn.running_var + bn.eps)
                beta = bn.weight
                gamma = bn.bias

                w_s = beta / var_sqrt
                b_o = -mean / var_sqrt * beta + gamma

            self.affine_1 = Affine(w_s.shape[0]).to(w_s.device)
            with torch.no_grad():
                self.affine_1.weight.set_(w_s)
                self.affine_1.bias.set_(b_o)


    def dump_params(self, quantized):
        params = {}
        params['affine_1'] = gen_param_dict(self.affine_1, quantized)
        params['linear'] = gen_param_dict(self.layers, quantized)
        params['affine_2'] = gen_param_dict(self.affine_2, quantized)
        params['res_affine'] = gen_param_dict(self.res_affine, quantized)
        return params


class CrossChannelSublayer(nn.Module):
    def __init__(self, in_patches, in_channels, layerscale_init=0.1, hidden_factor=4):
        super().__init__()

        self.affine_1 = ModifiedBatchNorm1d(in_channels)
        self.affine_2 = Affine(in_channels, alpha_init=layerscale_init)

        h = int(in_channels*hidden_factor)
        self.layers = nn.Sequential(
            nn.Linear(in_channels, h),
            _ACT_FN(),
            nn.Linear(h, in_channels),
        )

        # A helper layer to absorb affines
        self.res_affine = None


    def forward(self, x):
        if not hasattr(self, 'res_affine') or self.res_affine is None:
            x_res = x
        else:
            x_res = self.res_affine(x)


        x = self.affine_1(x)
        x = self.layers(x)
        x = self.affine_2(x)

        x = x + x_res
        return x


    def batchnorm_to_affine(self):
        if type(self.affine_1) != Affine:
            with torch.no_grad():
                bn = self.affine_1.bn
                mean = bn.running_mean
                var_sqrt = torch.sqrt(bn.running_var + bn.eps)
                beta = bn.weight
                gamma = bn.bias

                w_s = beta / var_sqrt
                b_o = -mean / var_sqrt * beta + gamma

            self.affine_1 = Affine(w_s.shape[0]).to(w_s.device)
            with torch.no_grad():
                self.affine_1.weight.set_(w_s)
                self.affine_1.bias.set_(b_o)


    def dump_params(self, quantized):
        params = {}
        params['linear_1'] = gen_param_dict(self.layers[0], quantized)
        params['linear_2'] = gen_param_dict(self.layers[2], quantized)
        params['res_affine'] = gen_param_dict(self.res_affine, quantized)
        return params


class ResMlpBlock(nn.Module):
    def __init__(self, in_patches, in_channels, hidden_factor=4):
        super().__init__()

        self.layers = nn.Sequential(
            CrossPatchSublayer(in_patches, in_channels),
            CrossChannelSublayer(in_patches, in_channels, hidden_factor=hidden_factor)
        )


    def forward(self, x):
        return self.layers(x)


    def absorb_affines(self):
        for m in self.layers:
            if hasattr(m, 'absorb_affines'):
                m.absorb_affines()


class ResMLP(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, img_size=32, patch_size=4,
            embed_dim=96, hidden_factor=4, nlayers=4):

        super().__init__()
        self.patch = img_size//patch_size
        self.patch_size = patch_size

        f = (self.patch_size)**2 * in_channels
        d = self.patch**2

        self.embed_linear = nn.Linear(f, embed_dim)

        layers = [ ResMlpBlock(d, embed_dim, hidden_factor=hidden_factor) for _ in range(nlayers) ]
        self.layers = nn.Sequential(*layers)

        self.affine = ModifiedBatchNorm1d(embed_dim)
        self.pool = AvgPool(1, d)
        self.classifier = nn.Linear(embed_dim, out_channels)


    def patch_gen(self, x):
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        out = out.permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.patch**2, -1).contiguous()
        return out


    def forward(self, x, early_exit_after_block=None):
        batch_size = x.shape[0]
        x = self.patch_gen(x)
        x = self.embed_linear(x)

        if early_exit_after_block is not None:
            for i in range(early_exit_after_block):
                x = self.layers[i](x)
            return x

        x = self.layers(x)
        x = self.affine(x)
        # x = x.mean(dim=1).reshape(batch_size, -1)
        x = self.pool(x)
        x = self.classifier(x)
        return x


    def batchnorm_to_affine(self):
        if type(self.affine) != Affine:
            with torch.no_grad():
                bn = self.affine.bn
                mean = bn.running_mean
                var_sqrt = torch.sqrt(bn.running_var + bn.eps)
                beta = bn.weight
                gamma = bn.bias

                w_s = beta / var_sqrt
                b_o = -mean / var_sqrt * beta + gamma

            self.affine = Affine(w_s.shape[0]).to(w_s.device)
            with torch.no_grad():
                self.affine.weight.set_(w_s)
                self.affine.bias.set_(b_o)

        for m in self.layers.modules():
            if hasattr(m, 'batchnorm_to_affine'):
                m.batchnorm_to_affine()

    # Pre-Quantization step to conver batchnorms and aborb affines
    # also adds a scalar layer in residual path
    # so addition occurs with same scale
    def absorb_affines(self):
        device = self.embed_linear.weight.device

        # Convert BatchNorms to Affine
        # Simplifies code for absorbing affines in this next part
        self.batchnorm_to_affine()

        # ------------------------------------------------------------------------
        # Absorb the affine before the classifier into classifier
        with torch.no_grad():
            with torch.no_grad():
                w_s = self.affine.weight
                b_o = self.affine.bias
                b_o = torch.matmul(b_o, self.classifier.weight.T)

                self.classifier.weight.mul_(w_s.view(1, -1))
                self.classifier.bias.add_(b_o)
        self.affine = nn.Identity()


        # ------------------------------------------------------------------------
        # Absorb the affines in the CrossChannelSublayer (not the residual path)
        for i in range(len(self.layers)):
            m = self.layers[i].layers[1]            # This Block's CrossChannelSublayer

            with torch.no_grad():
                w_s = m.affine_1.weight
                b_o = m.affine_1.bias
                b_o = torch.matmul(b_o, m.layers[0].weight.T)

                m.layers[0].weight.mul_(w_s.view(1, -1))
                m.layers[0].bias.add_(b_o)

            m.affine_1 = nn.Identity()

            with torch.no_grad():
                w_s = m.affine_2.weight
                b_o = m.affine_2.bias

                m.layers[2].weight.mul_(w_s.view(-1, 1))
                m.layers[2].bias.mul_(w_s)
                m.layers[2].bias.add_(b_o)

            m.affine_2 = nn.Identity()

        # ------------------------------------------------------------------------
        # Add empty affines in the residual paths
        # this to rescale the path to have the same scalar during residual addition
        for i in range(len(self.layers)):
            m = self.layers[i].layers[0]
            dim = m.affine_1.weight.shape[0]
            m.res_affine = Affine(dim).to(device)

            m = self.layers[i].layers[1]
            m.res_affine = Affine(dim).to(device)


    # Matches scalers between layers to ensure no rescaling takes place
    def match_abs_maxes(self):
        # ------------------------------------------------------------------------
        # embed_linear to first block
        equalize_scales(
            self.embed_linear.output_abs_max,
            self.layers[0].layers[0].affine_1.input_abs_max,
        )

        # ------------------------------------------------------------------------
        # within each block
        for i in range(len(self.layers)):
            m0 = self.layers[i].layers[0]
            m1 = self.layers[i].layers[1]
            equalize_scales(
                m0.affine_1.output_abs_max,
                m0.layers.input_abs_max
            )
            equalize_scales(
                m0.layers.output_abs_max,
                m0.affine_2.input_abs_max
            )
            equalize_scales(
                m0.affine_2.output_abs_max,
                m1.layers[0].input_abs_max,
            )
            equalize_scales(
                m1.layers[0].output_abs_max,
                m1.layers[2].input_abs_max
            )

        # ------------------------------------------------------------------------
        # b/w each block
        for i in range(len(self.layers))[:-1]:
            equalize_scales(
                self.layers[i].layers[1].layers[2].output_abs_max,
                self.layers[i+1].layers[0].affine_1.input_abs_max
            )

        # ------------------------------------------------------------------------
        # b/w last block to classifier
        equalize_scales(
            self.layers[-1].layers[1].layers[2].output_abs_max,
            self.pool.input_abs_max,
            self.pool.output_abs_max,
            self.classifier.input_abs_max
        )

        # ------------------------------------------------------------------------
        # set the affines in residual path properly
        # match it scales, and set the weight to ratio for the rescaling
        for i in range(len(self.layers)):
            m0 = self.layers[i].layers[0]
            m1 = self.layers[i].layers[1]

            # scale_ratio = m0.affine_2.output_abs_max / m0.res_affine.output_abs_max
            # with torch.no_grad():
            #     m0.res_affine.weight.fill_(scale_ratio)
            with torch.no_grad():
                m0.res_affine.input_abs_max.copy_(m0.affine_1.input_abs_max)
                m0.res_affine.output_abs_max.copy_(m0.affine_2.output_abs_max)

            with torch.no_grad():
                m1.res_affine.input_abs_max.copy_(m1.layers[0].input_abs_max)
                m1.res_affine.output_abs_max.copy_(m1.layers[2].output_abs_max)


    def dump_params(self, quantized=False):
        params = {}
        params['embed_linear'] = gen_param_dict(self.embed_linear, quantized=quantized)
        params['blocks'] = [
            {
                'token_mixer': m.layers[0].dump_params(quantized=quantized),
                'mlp': m.layers[1].dump_params(quantized=quantized)
            }
            for m in self.layers
        ]
        params['classifier_linear'] = gen_param_dict(self.classifier, quantized=quantized)
        return params
