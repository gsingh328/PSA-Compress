#ifndef PROCESSOR_H
#define PROCESSOR_H

#include "accel_common.h"

// #define USE_DSP_PACKING


void processor(
	// Linear Params
	accel_io_t io_buff[IO_BUFF_N][IO_BUFF_DIM_0][IO_BUFF_DIM_1],
	accel_io_t res_buff[IO_BUFF_DIM_0][IO_BUFF_DIM_1],
	const dim_t io_x_idx,
	const dim_t io_y_idx,
	const accel_w_t w[W_BUFF_DIM_0][W_BUFF_DIM_1],
	const accel_b_t b[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	const accel_s_t s[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	const dim_t x_dim_0,
	const dim_t x_dim_1,
	const dim_t w_dim_1,
	const bool add_bias,
	const bool apply_relu,
	const bool add_residual,
	const bool transpose_input,
	const bool transpose_output,
	const bool write_res,

	// Affine Params
	const bool apply_pre_affine,
	const accel_w_t w_pre_aff[W_AFF_BUFF_DIM_0][W_AFF_BUFF_DIM_1],
	const accel_b_t b_pre_aff[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	const accel_s_t s_pre_aff[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	const bool apply_post_affine,
	const accel_w_t w_post_aff[W_AFF_BUFF_DIM_0][W_AFF_BUFF_DIM_1],
	const accel_b_t b_post_aff[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	const accel_s_t s_post_aff[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	const bool apply_res_affine,
	const accel_w_t w_res_aff[W_AFF_BUFF_DIM_0][W_AFF_BUFF_DIM_1],
	const accel_b_t b_res_aff[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	const accel_s_t s_res_aff[BS_BUFF_DIM_0][BS_BUFF_DIM_1]
);

#endif // PROCESSOR_H
