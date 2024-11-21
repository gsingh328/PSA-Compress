import torch
from torch.nn.parameter import Parameter

from .psa import psa_func
from .psa_base import PersonalSelfAttentionBase
from .lerp import Lerp_Func


# Custom autograd implementation of PersonalSelfAttention
# Here we don't compute any gradients of the input
# Since they will belong to a buffer and not a Parameter
# Saves about ~10-20% memory compared to allowing Pytorch handling it
class GridPSA_Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid, v_weight, q_weight, q_bias, o_weight, o_bias,
            epsilon, save_grid_outputs):

        grid_attn, grid_v, grid_nl_o_pre_aff, grid_nl_o = psa_func(
            grid, v_weight, q_weight, q_bias, o_weight, o_bias, epsilon)

        ctx.save_grid_outputs = save_grid_outputs
        ctx.using_aff_out_bias = o_bias is not None

        if save_grid_outputs:
            ctx.save_for_backward(
                q_weight, q_bias, epsilon,
                o_weight,
                grid_attn, grid_v, grid_nl_o_pre_aff, grid)
        else:
            ctx.save_for_backward(
                v_weight, q_weight, q_bias, epsilon,
                o_weight, o_bias,
                grid)

        return grid_nl_o

    @staticmethod
    def backward(ctx, grad_out):
        save_grid_outputs = ctx.save_grid_outputs
        using_aff_out_bias = ctx.using_aff_out_bias
        if save_grid_outputs:
            (q_weight, q_bias, epsilon,
            o_weight,
            grid_attn, grid_v, grid_nl_o_pre_aff, grid) = ctx.saved_tensors
        else:
            (v_weight, q_weight, q_bias, epsilon,
            o_weight, o_bias,
            grid) = ctx.saved_tensors

            # Recalculate the grid PSA outputs
            grid_attn, grid_v, grid_nl_o_pre_aff, grid_nl_o = psa_func(
                grid, v_weight, q_weight, q_bias, o_weight, o_bias, epsilon)
            grid_nl_o_pre_aff = grid_nl_o.clone()
            if o_weight is not None:
                grid_nl_o.mul_(o_weight)
            if o_bias is not None:
                grid_nl_o.add_(o_bias)

        grid_grad_out = grad_out

        # For input gradients accumulate across the last dimension (ie, the expansion)
        # in_accum_dims = -1

        # For our learnable parameters, accumulate across all dimensions except the last
        param_accum_dims = tuple(range(len(grid.shape) - 1))

        # Backwards across the affine operation
        aff_grad = grid_grad_out
        o_weight_grad = None
        o_bias_grad = None
        if o_weight is not None:
            aff_grad = grid_grad_out * o_weight
            o_weight_grad = (grid_grad_out * grid_nl_o_pre_aff).sum(dim=param_accum_dims)
        if using_aff_out_bias:
            o_bias_grad = grid_grad_out.sum(dim=param_accum_dims)

        # Backwards across output calculation from value projection and attention
        v_grad = aff_grad.unsqueeze(-1) * grid_attn
        attn_grad = aff_grad.unsqueeze(-1) * grid_v

        # Backwards across value projection
        # grid_v_grad = (v_grad * v_weight).sum(dim=in_accum_dims)
        v_weight_grad = (v_grad * grid.unsqueeze(-1)).sum(dim=param_accum_dims)

        # Backwards across the softmax (don't forget about the negative in score)
        q_grad = -grid_attn * (attn_grad - torch.sum(attn_grad * grid_attn, dim=-1, keepdim=True))

        # Backwards across the score calculation
        q_weight_sq_eps = (q_weight ** 2) + epsilon
        diff = grid.unsqueeze(-1) - q_bias
        dout_dx = 2 * diff / q_weight_sq_eps
        dout_dq_bias = -dout_dx
        dout_dq_weight = -2 * (diff**2) * q_weight / (q_weight_sq_eps**2)

        # Multiply by the incoming gradient for score calculation
        # grid_q_grad = (q_grad * dout_dx).sum(dim=in_accum_dims)
        q_bias_grad = (q_grad * dout_dq_bias).sum(dim=param_accum_dims)
        q_weight_grad = (q_grad * dout_dq_weight).sum(dim=param_accum_dims)

        # Add the backward gradients of the input from value projection and score calculation
        # grid_grad = grid_v_grad + grid_q_grad

        # NOTE: We dont depend on the backward propogation from grid_grad to get gradients of x
        # since we already store the slopes (ie gradient) of the grid from the forward call
        # These are applied in the Lerp
        x_grad = None

        # Assuming Epsilon is scalar with no learnable gradient
        epsilon_grad = None

        return (x_grad, v_weight_grad, q_weight_grad, q_bias_grad, o_weight_grad, o_bias_grad,
            epsilon_grad, None)


class LerpPersonalSelfAttention(PersonalSelfAttentionBase):
    def __init__(self, *args, grid_size=2**5+1, grid_range=4,
            save_memory=True,  save_grid_outputs=True,
            max_norm_grid_range=False,
            max_norm_per_feature=False,
            use_custom_autograd=True,
            corrected_grid_init=False,
            **kwargs):

        super().__init__(*args, **kwargs)

        # if grid_size % 2 == 0:
        #     print('WARNING: Even number of sample points in grid is not supported.'
        #         ' Adding +1 to grid_size')
        #     grid_size += 1

        self.grid_size = grid_size
        self.grid_range = grid_range
        self.save_memory = save_memory
        self.save_grid_outputs = save_grid_outputs
        self.max_norm_grid_range = max_norm_grid_range
        self.max_norm_per_feature = max_norm_per_feature
        self.use_custom_autograd = use_custom_autograd

        if not self.max_norm_grid_range and self.max_norm_per_feature:
            print('WARNING: max_norm_per_feature is ignored if max_norm_grid_range==False')

        # correct grid in-case of even grid size
        # basically ensures a grid point at exactly 0.0
        if corrected_grid_init:
            grid = torch.linspace(
                -grid_range, grid_range - (2*grid_range/grid_size), self.grid_size).view(-1, 1)
        else:
            grid = torch.linspace(
                -grid_range, grid_range, self.grid_size).view(-1, 1)

        self.register_buffer('grid', torch.tile(
            grid, (1, self.input_features)))

        # Store offsets required while indexing
        ifeat_offset = torch.arange(0, self.input_features)
        grid_ifeat_offset = (ifeat_offset * self.grid_size).long()
        self.register_buffer('grid_ifeat_offset', grid_ifeat_offset.reshape(1, -1))
        self.grid_nl_o = None

    def extra_repr(self):
        s = ('input_features={input_features}, '
             'expansion_features={expansion_features}, '
             'grid_size={grid_size}, '
             'grid_range={grid_range}, '
             'max_norm_grid_range={max_norm_grid_range}, '
             'max_norm_per_feature={max_norm_per_feature}, '
             'use_custom_autograd={use_custom_autograd}')
        return s.format(**self.__dict__) + f', epsilon={self.epsilon.item()}'

    def forward(self, x):
        if self.max_norm_grid_range:
            with torch.no_grad():
                if self.max_norm_per_feature:
                    in_abs_max, _ = torch.max(x.abs(), dim=0, keepdim=True)
                    in_abs_max, _ = torch.max(in_abs_max, dim=1, keepdim=True)
                else:
                    in_abs_max = x.abs().max().item()

                grid_scalar = in_abs_max / self.grid_range
                grid = self.grid * grid_scalar
                grid_range = in_abs_max
        else:
            grid = self.grid
            grid_range = self.grid_range

        # If training make sure to recalculate the grid lut outputs
        if self.training:
            self.grid_nl_o = None

        if self.grid_nl_o is None:
            if self.use_custom_autograd:
                self.grid_nl_o = GridPSA_Func.apply(
                    grid, self.v_weight, self.q_weight, self.q_bias,
                    self.o_weight, self.o_bias, self.epsilon, self.save_grid_outputs)
            else:
                _, _, _, self.grid_nl_o = psa_func(
                    grid, self.v_weight, self.q_weight, self.q_bias,
                    self.o_weight, self.o_bias, self.epsilon)

        lerp_output = Lerp_Func.apply(
            x, self.grid_nl_o, self.grid_size, grid_range, self.grid_ifeat_offset,
            self.save_memory
        )

        # If training we can discard the grid lut outputs since they will be
        # updated at each epoch
        if self.training:
            self.grid_nl_o = None

        return lerp_output
