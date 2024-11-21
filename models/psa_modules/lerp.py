import torch


def calc_fracs_indices(x, grid_size, grid_range):
    step_size = (grid_size - 1) / (2 * grid_range)

    # One less index at max so we can gaurantee
    # indices are within bounds after the grid_nl_o.diff() call
    tmp_x = x + grid_range
    tmp_x.mul_(step_size)
    tmp_x_floor = tmp_x.floor()
    tmp_x_floor.clamp_(0, grid_size - 2)

    indices = tmp_x_floor.long()
    fracs = tmp_x - tmp_x_floor

    return fracs, indices


class Lerp_Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grid_nl_o, grid_size, grid_range, grid_ifeat_offset,
            save_memory):
        fracs, indices = calc_fracs_indices(x, grid_size, grid_range)

        # Flatten our base Lookup Tables
        # make sure to put the nonlinear output as the last axis
        grid_nl_o_shape = grid_nl_o.shape
        grid_nl_o = grid_nl_o.swapaxes(-1, -2).reshape(-1)
        grid_nl_o_slopes = grid_nl_o.diff()

        # Add custom offest per input feature
        # This way each feature indexes a seperate section within the lookup table
        indices.add_(grid_ifeat_offset)

        # LERP
        y = grid_nl_o[indices] + fracs * grid_nl_o_slopes[indices]

        # Save scalars for backward call
        ctx.grid_size = grid_size
        ctx.grid_range = grid_range
        ctx.grid_shape = grid_nl_o_shape
        ctx.save_memory = save_memory

        # If saving memory, we will recompute indices and fracs from input x
        if save_memory:
            ctx.save_for_backward(x, grid_nl_o_slopes, grid_ifeat_offset)
        else:
            ctx.save_for_backward(indices, fracs, grid_nl_o_slopes)

        return y

    @staticmethod
    def backward(ctx, grad_out):
        grid_size = ctx.grid_size
        grid_range = ctx.grid_range
        grid_shape = ctx.grid_shape
        save_memory = ctx.save_memory
        step_size = (grid_size - 1) / (2 * grid_range)
        if save_memory:
            x, grid_nl_o_slopes, grid_ifeat_offset = ctx.saved_tensors

            # Recalculate our indices and fractions
            fracs, indices = calc_fracs_indices(x, grid_size, grid_range)
            indices.add_(grid_ifeat_offset)
        else:
            indices, fracs, grid_nl_o_slopes = ctx.saved_tensors

        # Use the precalculated slopes from forward call to generate
        # gradients in backward call
        grid_nl_o_slopes.mul_(step_size)
        x_grad = grid_nl_o_slopes[indices] * grad_out

        # Accumulate gradients into each grid point relative to their distance
        grid_grad_out = torch.zeros(grid_shape.numel(),
            device=grid_nl_o_slopes.device)

        # right indexing gradients
        grid_grad_out.scatter_add_(0,
            indices.reshape(-1) + 1, (fracs * grad_out).reshape(-1))

        # left indexing gradients
        grid_grad_out.scatter_add_(0,
            indices.reshape(-1), ((1 - fracs) * grad_out).reshape(-1))

        # indices assumes that grid_grad_out is of shape (i_features, grid_size)
        # but going backwards across PSA we need it to be transposed
        grid_grad_out = grid_grad_out.view(
            grid_shape[-1], grid_shape[-2])
        grid_grad_out = (grid_grad_out.swapaxes(0, 1)
            .view(grid_shape).contiguous())

        return (x_grad, grid_grad_out,
            None, None, None, None, None)
