# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import math


def get_Qn_Qp(bits, signed=True):
    if signed:
        # 2-bits is assumed to imply ternary networks (-1, 0, 1)
        if bits <= 2:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (bits - 1)
            Qp = 2 ** (bits - 1) - 1
    else:
        Qn = 0
        Qp = 2 ** (bits) - 1
    return Qn, Qp


class ElasticQuantBinarizerSigned(torch.autograd.Function):
    """
        Modified from Learned Step-size Quantization.
        https://arxiv.org/abs/1902.08153
    """
    @staticmethod
    def forward(ctx, input, alpha, num_bits):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :return: quantized output
        """

        ctx.num_bits = num_bits
        if num_bits == 32:
            return input

        Qn, Qp = get_Qn_Qp(num_bits, signed=True)

        eps = torch.tensor(0.00001).float().to(alpha.device)
        alpha = torch.where(alpha > eps, alpha, eps)

        assert (alpha > 0).all(), 'alpha = {:.6f} becomes non-positive'.format(alpha)

        if not Qp:
            grad_scale = 1.0 / math.sqrt(input.numel())
        else:
            grad_scale = 1.0 / math.sqrt(input.numel() * Qp)

        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (input / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        if ctx.num_bits == 1:
            grad_alpha = ((input_.sign()) * grad_output * grad_scale)
        else:
            grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                    -q_w + q_w.round())) * grad_output * grad_scale)

        if alpha.shape == grad_alpha.shape:
            # Dont accumulate if we already match shapes
            pass
        elif torch.numel(alpha) > 1:
            # If alpha is not a scalar, accumulate on the batch dimension (ie the first one)
            grad_alpha = grad_alpha.sum(dim=0)
        else:
            # Alpha is scalar, accumulate on all dimensions
            grad_alpha = grad_alpha.sum().unsqueeze(dim=0)

        grad_alpha = grad_alpha.reshape_as(alpha)
        assert grad_alpha.shape == alpha.shape, \
            f"alpha gradient mismatched shape {list(grad_alpha.shape)} to {list(alpha.shape)} " + \
            f"{grad_alpha} != {alpha}"

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None
