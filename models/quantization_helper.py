import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from typing import Tuple
import math
from typing import Optional


from torch.utils.data import DataLoader

from models.psa_modules.respsa import ResPSA
from models.psa_modules.lerp_psa import LerpPersonalSelfAttention
from models.utils import Affine, AvgPool

from tqdm import tqdm
import numpy as np

from .quant_utils import ElasticQuantBinarizerSigned


LINEAR_REQUANT_METHOD = 'SHFT'
assert(LINEAR_REQUANT_METHOD in ['FLOAT_DIV', '32_MUL', '16_SPLIT_MUL', 'SHFT'])

ADD_RND_FACTOR=True
SATURATE_OUTPUT=True

# =========================================================================


def report_diff(x, y, header=''):
    pass
    # x, y = x.astype(np.float32), y.astype(np.float32)
    # err = x - y
    # err_abs = np.abs(err)

    # should_print = not (err_abs < 2).all()
    # if should_print:
    #     print('=' * 30, header, '=' * 30)
    #     print('Error relative to FP Fake Quantize')
    #     print(f'Matched with error <= 1: {(err_abs < 2).all()}')
    #     print(f'Error Mean = {np.mean(err):.4f}\n'
    #         f'STD = {np.std(err):.4f}\n'
    #         f'Percentile (50, 75, 90, 99) = ({np.percentile(err_abs, 50):.0f}, {np.percentile(err_abs, 75):.0f}, '
    #         f'{np.percentile(err_abs, 90):.0f}, {np.percentile(err_abs, 99):.0f})\n'
    #         f'Abs Max = {np.max(err_abs):.0f}')
    # return np.max(err_abs)


def get_imax_scale(x_max, x_bits):
    n = 2 ** (x_bits - 1)
    s = n / x_max
    return n, s


def quantize_symetric(x, n, s, dequantize=True):
    x_clip_min = -n / s
    x_clip_max = (n - 1) / s

    # Do clipping with gradients
    x = x.clip(x_clip_min, x_clip_max)

    # Do actual quantization with no gradients
    with torch.no_grad():
        x_rnd = (x * s).round()

        if not dequantize:
            return x_rnd.type(torch.int32), s

        x_rnd = x_rnd / s
        x.copy_(x_rnd)
    return x, s


def quant_forward(x, x_max, x_bits, dequantize=True, transpose=False):
    if x is None:
        return None, 1.0

    n, s = get_imax_scale(x_max, x_bits)
    s = s.to(x.device)

    # return quantize_symetric(x, n, s, dequantize=dequantize)
    alpha = (1./s)
    if len(x.shape) == 1:
        alpha = alpha.reshape(-1)

    xq = ElasticQuantBinarizerSigned.apply(
        x if not transpose else x.transpose(-1, -2),
        alpha if not transpose else alpha.transpose(-1, -2),
        x_bits
    )
    if transpose:
        xq = xq.transpose(-1, -2)

    if not dequantize:
        with torch.no_grad():
            xq = (xq * s).round().type(torch.int32)

    assert xq.shape == x.shape, f"quantization changed shape {list(xq.shape)} to {list(x.shape)}"
    return xq, s


def quant_forward_ste(x, x_max, x_bits, dequantize=True, transpose=False):
    if x is None:
        return None, 1.0

    n, s = get_imax_scale(x_max, x_bits)
    s = s.to(x.device)
    return quantize_symetric(x, n, s, dequantize=dequantize)


def recalc_w_abs_max_for_shift(input_abs_max, input_bits,
        weight_abs_max, weight_bits, output_abs_max, output_bits, rnd_s=True):

    if LINEAR_REQUANT_METHOD != 'SHFT':
        return weight_abs_max, None

    _, x_s = get_imax_scale(input_abs_max, input_bits)
    _, w_s = get_imax_scale(weight_abs_max, weight_bits)
    _, y_s = get_imax_scale(output_abs_max, output_bits)

    yq_s = (x_s * w_s) / y_s

    if rnd_s:
        requant_s = torch.log2(yq_s).round()
    else:
        requant_s = torch.log2(yq_s).floor()

    target_yq_s = 2.**(requant_s)
    target_w_s = (target_yq_s * y_s) / x_s
    target_weight_abs_max = 2 ** (weight_bits - 1) / target_w_s
    assert target_weight_abs_max.shape == weight_abs_max.shape
    return target_weight_abs_max, requant_s.reshape(-1)


def quantizer_range_observer(model, input, output):
    if model.quant_observe:
        for x in input:
            x_abs = x.data.abs()
            x_max, _ = x_abs.max(dim=-1)
            while len(x_max.shape) > 1:
                x_max, _ = x_max.max(dim=-1)

            if model.input_abs_maxes is None:
                model.input_abs_maxes = x_max
            else:
                model.input_abs_maxes = torch.cat([model.input_abs_maxes, x_max])

        y_abs = output.data.abs()
        y_max, _ = y_abs.max(dim=-1)
        while len(y_max.shape) > 1:
            y_max, _ = y_max.max(dim=-1)

        if model.output_abs_maxes is None:
            model.output_abs_maxes = x_max
        else:
            model.output_abs_maxes = torch.cat([model.output_abs_maxes, y_max])


def requant(self, yq, yq_s, method=LINEAR_REQUANT_METHOD):
    if method == 'FLOAT_DIV':
        # s = (x_s * w_s).view(-1)

        # yqf = yq.float() / s

        # yq, _ = quant_forward(
        #     yqf, self.output_abs_max, self.output_bits,
        #     dequantize=False)

        yq = (yq.float() / yq_s.round()).round()

    elif method in ['32_MUL', '16_SPLIT_MUL']:
        n = 32 - 8
        n2 = 32
        yq_s = torch.round(2.**n / yq_s).clamp(0, 2**n2 - 1).type(torch.int32)

        if method == '32_MUL':
            yq = (yq.type(torch.int64) * yq_s.type(torch.int64)) >> n
            yq = yq.type(torch.int32)
        else:
            yq_1 = torch.floor_divide(yq, 2.**16).type(torch.int32)
            yq_2 = torch.remainder(yq, 2.**16).type(torch.int32)
            yq_s_1 = torch.floor_divide(yq_s, 2.**16).type(torch.int32)
            yq_s_2 = torch.remainder(yq_s, 2.**16).type(torch.int32)

            frac_1 = torch.round((yq_1 * yq_s_1) * 2.**8).type(torch.int32)
            frac_2 = torch.round((yq_1 * yq_s_2) / 2.**8).type(torch.int32)
            frac_3 = torch.round((yq_2 * yq_s_1) / 2.**8).type(torch.int32)
            frac_4 = torch.round((yq_2 * yq_s_2) / 2.**24).type(torch.int32)
            frac_4 = torch.round(
                (torch.round(yq_2 / 2.**8).type(torch.int32) *
                torch.round(yq_s_2 / 2.**8).type(torch.int32)
                ) / 2.**8).type(torch.int32)

            yq = frac_1 + frac_2 + frac_3 + frac_4
            # yq = frac_2 + frac_3 + frac_4
    elif method == 'SHFT':
        # yq_s_l2 = (torch.log2(yq_s).round()).type(torch.int32)
        yq_s_l2 = self.requant_s.to(yq.device)

        yq = yq >> (yq_s_l2 - 1)

        rnd_factor = 0
        if ADD_RND_FACTOR:
            rnd_factor = torch.bitwise_and(yq, 1)

        yq = (yq >> 1) + rnd_factor
        # yq = (yq / 2.**yq_s_l2).round().type(torch.int32)
    else:
        raise NotImplementedError

    return yq


def qlinear_forward(self, x):
    if not self.quant_apply:
        return self.native_forward(x)

    # weight_abs_max = self.weight.abs().max(dim=1)[0].view(-1, 1)
    weight_abs_max = self.weight_abs_max
    target_weight_abs_max, _ = recalc_w_abs_max_for_shift(
        self.input_abs_max, self.input_bits,
        weight_abs_max, self.weight_bits,
        self.output_abs_max, self.output_bits)

    # Do default FP based quantization
    # This will properly handle STE gradients needed for training
    x, x_s = quant_forward(x, self.input_abs_max, self.input_bits)
    w, w_s = quant_forward(self.weight, target_weight_abs_max, self.weight_bits, transpose=True)
    b, b_s = quant_forward(self.bias, self.output_abs_max, self.bias_bits)

    y = F.linear(x, w, b)

    y, y_s = quant_forward(y, self.output_abs_max, self.output_bits)

    if self.use_int_ops:
        # print("Linear - !!")
        with torch.no_grad():
            xq = (x * x_s).round().cpu().type(torch.int32)
            x_s = x_s.cpu()

            wq = (w * w_s).round().cpu().type(torch.int32)
            w_s = w_s.cpu()

            bq = None
            if b is not None:
                bq = (b * b_s).round().cpu().type(torch.int32)
                b_s = b_s.cpu()

            yq = torch.matmul(xq, wq.T)

            y_s = y_s.cpu()
            yq_s = (x_s * w_s) / y_s
            # yq_s = yq_s.T
            yq_s = yq_s.view(-1)

            yq = requant(self, yq, yq_s)

            if bq is not None:
                yq = yq + bq

            n = 2**(self.output_bits - 1)
            if SATURATE_OUTPUT:
                yq = yq.clamp(-n, n-1)
            else:
                yq = yq.type(torch.int8)
            yqf = yq / y_s
            yqf = yqf.to(y.device)
            report_diff(y.cpu().numpy(), yqf.cpu().numpy(), header='Linear')
            y.copy_(yqf)
    return y


def qpsa_forward(self, x):
    if not self.quant_apply:
        return self.native_forward(x)

    # Do default FP based quantization
    # This will properly handle STE gradients needed for training
    xq, xq_s = quant_forward(x, self.input_abs_max, self.input_bits)

    y = self.native_forward(xq)

    yq, yq_s = quant_forward(y, self.output_abs_max, self.output_bits)
    return yq


def affine_forward(self, x):
    if not self.quant_apply:
        return self.native_forward(x)

    # weight_abs_max = self.weight.abs().view(-1)
    weight_abs_max = self.weight_abs_max
    target_weight_abs_max, _ = recalc_w_abs_max_for_shift(
        self.input_abs_max, self.input_bits,
        weight_abs_max, self.weight_bits,
        self.output_abs_max, self.output_bits)

    # Do default FP based quantization
    # This will properly handle STE gradients needed for training
    x, x_s = quant_forward(x, self.input_abs_max, self.input_bits)
    w, w_s = quant_forward(self.weight, target_weight_abs_max, self.weight_bits)
    b, b_s = quant_forward(self.bias, self.output_abs_max, self.bias_bits)

    y = x * w + b

    y, y_s = quant_forward(y, self.output_abs_max, self.output_bits)

    if self.use_int_ops:
        # print("Affine - !!")
        with torch.no_grad():
            xq = (x * x_s).round().cpu().type(torch.int32)
            x_s = x_s.cpu()

            wq = (w * w_s).round().cpu().type(torch.int32)
            w_s = w_s.cpu()

            bq = None
            if b is not None:
                bq = (b * b_s).round().cpu().type(torch.int32)
                b_s = b_s.cpu()

            yq = xq * wq

            y_s = y_s.cpu()
            yq_s = (x_s * w_s) / y_s
            # yq_s = yq_s.T
            yq_s = yq_s.view(-1)

            yq = requant(self, yq, yq_s)

            if bq is not None:
                yq = yq + bq

            n = 2**(self.output_bits - 1)
            if SATURATE_OUTPUT:
                yq = yq.clamp(-n, n-1)
            else:
                yq = yq.type(torch.int8)
            yqf = yq / y_s
            yqf = yqf.to(y.device)
            report_diff(y.cpu().numpy(), yqf.cpu().numpy(), header='Affine')
            y.copy_(yqf)
    return y


def avgpool_forward(self, x):
    if not self.quant_apply:
        return self.native_forward(x)

    # Do default FP based quantization
    # This will properly handle STE gradients needed for training
    x, x_s = quant_forward(x, self.input_abs_max, self.input_bits)

    y = x.sum(dim=self.dim) / self.n

    y, y_s = quant_forward(y, self.output_abs_max, self.output_bits)

    if self.use_int_ops:
        # print("AvgPool - !!")
        with torch.no_grad():
            xq = (x * x_s).round().cpu().type(torch.int32)
            x_s = x_s.cpu()

            # n_l2 = (torch.log2(torch.FloatTensor((self.n)).to(x.device)).round()).type(torch.int32)
            yq_s_l2 = int(round(math.log2(self.n)))
            yq = xq.sum(dim=self.dim)

            yq = yq >> (yq_s_l2 - 1)

            rnd_factor = 0
            if ADD_RND_FACTOR:
                rnd_factor = torch.bitwise_and(yq, 1)

            yq = (yq >> 1) + rnd_factor

            n = 2**(self.output_bits - 1)
            if SATURATE_OUTPUT:
                yq = yq.clamp(-n, n-1)
            else:
                yq = yq.type(torch.int8)
            yq = yq.to(y.device)
            y_s = y_s.to(y.device)
            yqf = yq / y_s
            yqf = yqf
            report_diff(y.cpu().numpy(), yqf.cpu().numpy(), header='AvgPool')
            y.copy_(yqf)
    return y


# =========================================================================

quant_layer_id = 0
def setup_quant_observers(model, layers_to_observe, quant_config, device):
    assert device is not None

    global quant_layer_id
    for m in model.modules():
        # ensure we dont re-initialize a layer
        if type(m) in layers_to_observe and not hasattr(m, 'quant_layer_id'):
            m.quant_layer_id = quant_layer_id
            # print('Setting up Quant Layer with ID: [{}] : {}'.format(m.quant_layer_id, str(m)))
            quant_layer_id = quant_layer_id + 1

            m.register_parameter('input_abs_max', nn.Parameter(torch.tensor(1.)))
            m.register_parameter('output_abs_max', nn.Parameter(torch.tensor(1.)))

            if type(m) == nn.Linear:
                x = torch.ones(m.weight.shape[0], 1)
            elif type(m) == Affine:
                x = torch.ones(m.weight.shape[0])
            else:
                x = torch.tensor(1.)

            m.register_buffer('weight_abs_max', x)
            m.register_buffer('bias_abs_max', torch.tensor(1.))

            m.input_abs_maxes = None
            m.output_abs_maxes = None

            m.quant_observe = False
            m.quant_apply = False
            m.use_int_ops = False
            m.input_bits = quant_config['input_bits']
            m.weight_bits = quant_config['weight_bits']
            m.bias_bits = quant_config['bias_bits']
            m.output_bits = quant_config['output_bits']

            m.quant_range_obs = m.register_forward_hook(quantizer_range_observer)


def do_quant_observe(model, dataloader, split_size, layers_observed, disable_tqdm=False, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'


    break_i = split_size

    # Ensure we are observing our input
    for m in model.modules():
        if type(m) in layers_observed:
            m.quant_observe = True

    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(dataloader, total=break_i, disable=disable_tqdm)):
            if i >= break_i:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            model(inputs)

    for m in model.modules():
        if type(m) in layers_observed:
            m.input_abs_maxes, _ = torch.sort(m.input_abs_maxes)
            m.output_abs_maxes, _ = torch.sort(m.output_abs_maxes)

            # Disable our observers
            m.quant_observe = False


def setup_quant_params_from_percentile(model, layers_observed, percentile=90):
    for m in model.modules():
        if type(m) in layers_observed:
            n = len(m.input_abs_maxes) - 1
            idx = np.floor(n * percentile / 100).astype(np.int32)
            with torch.no_grad():
                m.input_abs_max.fill_(m.input_abs_maxes[idx])

            n = len(m.output_abs_maxes) - 1
            idx = np.floor(n * percentile / 100).astype(np.int32)
            with torch.no_grad():
                m.output_abs_max.fill_(m.output_abs_maxes[idx])


def setup_layer_quant_params(model, layers_observed,
        reset_weight_scaler=False, linear_rescale_as_shift=LINEAR_REQUANT_METHOD=='SHFT'):

    # Use floor instead of rounded if we are resetting our scalar
    # floor() works better for initialization, but round() is necessary
    # when doing QAT
    rnd_s = not reset_weight_scaler

    for m in model.modules():
        if type(m) in layers_observed:
            if type(m) == nn.Linear:
                # weight_abs_max = m.weight.abs().max()

                if reset_weight_scaler:
                    weight_abs_max = m.weight.abs().max(dim=1)[0].view(-1, 1)
                else:
                    weight_abs_max = m.weight_abs_max

                if linear_rescale_as_shift:
                    target_weight_abs_max, requant_s = recalc_w_abs_max_for_shift(
                        m.input_abs_max, m.input_bits,
                        weight_abs_max, m.weight_bits,
                        m.output_abs_max, m.output_bits,
                        rnd_s=rnd_s)
                    m.requant_s = requant_s.view(-1).type(torch.int32)
                else:
                    target_weight_abs_max = weight_abs_max

                with torch.no_grad():
                    if torch.numel(m.weight_abs_max) > 1:
                        assert m.weight_abs_max.shape == target_weight_abs_max.shape, \
                            f"{m.weight_abs_max.shape} =/= {target_weight_abs_max.shape}"
                        m.weight_abs_max.copy_(target_weight_abs_max)
                    else:
                        m.weight_abs_max.fill_(target_weight_abs_max)
                    m.bias_abs_max.fill_(m.output_abs_max)

            if type(m) == Affine:
                # weight_abs_max = m.weight.abs().max()

                # weight_abs_max = m.weight.abs().max(dim=0)[0]
                if reset_weight_scaler:
                    weight_abs_max = m.weight.abs().view(-1)
                else:
                    weight_abs_max = m.weight_abs_max

                if linear_rescale_as_shift:
                    target_weight_abs_max, requant_s = recalc_w_abs_max_for_shift(
                        m.input_abs_max, m.input_bits,
                        weight_abs_max, m.weight_bits,
                        m.output_abs_max, m.output_bits,
                        rnd_s=rnd_s)
                    m.requant_s = requant_s.view(-1).type(torch.int32)
                else:
                    target_weight_abs_max = weight_abs_max

                with torch.no_grad():
                    if torch.numel(m.weight_abs_max) > 1:
                        assert m.weight_abs_max.shape == target_weight_abs_max.shape, \
                            f"{m.weight_abs_max.shape} =/= {target_weight_abs_max.shape}"
                        m.weight_abs_max.copy_(target_weight_abs_max)
                    else:
                        m.weight_abs_max.fill_(target_weight_abs_max)
                    m.bias_abs_max.fill_(m.output_abs_max)


def setup_quantizer(model, layers_to_quantize, dataloader, split_size, quant_config,
        load_path=None, find_activation_ranges=True,
        do_int_ops=False, disable_tqdm=False, device=None):

    setup_quant_observers(model, layers_to_quantize, quant_config, device)

    if load_path is not None:
        print('Loading from custom path: {}'.format(load_path))
        tmp_dct = torch.load(load_path)
        model.load_state_dict(tmp_dct, strict=False)

    if find_activation_ranges:
        do_quant_observe(model, dataloader, split_size, layers_to_quantize, disable_tqdm=disable_tqdm, device=device)

    if do_int_ops:
        set_int_ops(model, layers_to_quantize, True)


def set_quant_params(model, layers_to_quantize, percentile=None, reset_weight_scaler=False):
    if percentile is not None:
        setup_quant_params_from_percentile(model, layers_to_quantize, percentile)

    if hasattr(model, 'match_abs_maxes'):
        model.match_abs_maxes()
    setup_layer_quant_params(model, layers_to_quantize, reset_weight_scaler=reset_weight_scaler)

    # Start applying quantization
    replace_forwards_in_layers(model, layers_to_quantize)
    for m in model.modules():
        if type(m) in layers_to_quantize:
            m.quant_apply = True

            # remove observer since we no longer need it
            m.quant_range_obs.remove()


def replace_forwards_in_layers(model, layers_to_quantize):
    for m in model.modules():
        if type(m) in layers_to_quantize:
            if type(m) == nn.Linear:
                if not hasattr(m, 'native_forward'):
                    m.native_forward = m.forward
                    m.forward = types.MethodType(qlinear_forward, m)
            if type(m) == ResPSA:
                if not hasattr(m, 'native_forward'):
                    m.native_forward = m.forward
                    m.forward = types.MethodType(qpsa_forward, m)
            if type(m) == Affine:
                if not hasattr(m, 'native_forward'):
                    m.native_forward = m.forward
                    m.forward = types.MethodType(affine_forward, m)
            if type(m) == AvgPool:
                if not hasattr(m, 'native_forward'):
                    m.native_forward = m.forward
                    m.forward = types.MethodType(avgpool_forward, m)


def set_int_ops(model, layers_to_quantize, val):
    for m in model.modules():
        if type(m) in layers_to_quantize and hasattr(m, 'use_int_ops'):
            m.use_int_ops = val
