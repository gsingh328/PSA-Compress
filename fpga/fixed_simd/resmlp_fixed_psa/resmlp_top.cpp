#include <iostream>
#include <math.h>
#include <stdlib.h>

#include "tb_helper.h"

#include "resmlp.h"
#include "resmlp_model.h"
#include "utils.h"
#include "math.h"

// #include "quant_params_v2.h"
// #include "quant_samples_v2.h"
// #include "quant_samples_extra_v2.h"

#include "quant_params.h"
#include "quant_samples.h"
#include "quant_samples_extra.h"

// Use this parameters when the kernel is the unoptimized processor_old.cpp
// #include "lerp_psa_params.h"

// Use this parameters when the kernel is the optimized processor_new.cpp
#include "lerp_psa_params_transposed.h"

// #define Y_DIM_0 64
// #define Y_DIM_1 96
// #define REFERENCE_OUTPUT(i,j) samples_token_mixer_affine_1[i][j]
// #define REFERENCE_OUTPUT(i,j) samples_token_mixer_linear_1[i][j]

// #define Y_DIM_0 96
// #define Y_DIM_1 64
// #define REFERENCE_OUTPUT(i,j) samples_inter_0[i][j]
// #define REFERENCE_OUTPUT(i,j) samples_token_mixer_res[i][j]
// #define REFERENCE_OUTPUT(i,j) samples_token_mixer_affine_2[i][j]
// #define REFERENCE_OUTPUT(i,j) samples_token_mixer_post[i][j]
// #define REFERENCE_OUTPUT(i,j) samples_mlp_linear_1_relu[i][j]
// #define REFERENCE_OUTPUT(i,j) samples_inter_1[i][j]
// #define REFERENCE_OUTPUT(i,j) samples_inter_2[i][j]
// #define REFERENCE_OUTPUT(i,j) samples_inter_4[i][j]
#define REFERENCE_OUTPUT(i,j) samples_output[i]

#define Y_DIM_0 RESMLP_OUT_DIM_0
#define Y_DIM_1 RESMLP_OUT_DIM_1

// #define HARDWARE_OUTPUT(arr,idx,i,j) arr[idx / PORT_VEC_SIZE][idx % PORT_VEC_SIZE]
#define HARDWARE_OUTPUT(arr,idx,i,j) arr[i][j]


template<typename T, uint32_t i_features, uint32_t o_features>
void random_init_linear(
	T &params
	) {

	random_init_mat<accel_w_t, i_features, o_features>(
		params.w, 0, (32.0/3), false);
	random_init_vec<accel_b_t, o_features>(
		params.b, 0, (16.0/3), false);
	random_init_vec<accel_s_t, o_features>(
		params.s, 8, 1, true);

	// For Scalar ensure it is atleast 1
	for (uint32_t i = 0; i < o_features; i++) {
		CLIP(params.s[i], 1, 24);
	}
}


// Here the 1st dimension of x and w are reduced
// Batch factor basically represents how to split the shared dimension (x_dim_0)
// for a batched matmul required in multi-headed attention (to calculate dot-product)
template<uint32_t x_dim_0, uint32_t x_dim_1, uint32_t w_dim_1>
void gold_linear(
	const accel_io_t *x,
	const accel_io_t *res,
	const accel_w_t *w,
	const accel_b_t *b,
	const accel_s_t *s,
	const bool add_bias,
	const bool apply_relu,
	const bool add_residual,
	accel_io_t *y
	) {

	for (uint32_t w_d1 = 0; w_d1 < w_dim_1; w_d1++) {
		for (uint32_t x_d1 = 0; x_d1 < x_dim_1; x_d1++) {
			accel_acc_t acc = 0;
			for (uint32_t x_d0 = 0; x_d0 < x_dim_0; x_d0++) {
				acc += (
					x[IDX2D(x_d0, x_d1, x_dim_1)] *
					w[IDX2D(x_d0, w_d1, w_dim_1)]
				);
			}

			ROUNDED_SHIFT(acc, s[w_d1]);
			acc += (add_bias) ? b[w_d1] : (accel_b_t) 0;

			// Residual
			acc += (add_residual) ? res[IDX2D(w_d1, x_d1, x_dim_1)] : ((accel_io_t) 0);

			CLIP(acc, ACCEL_IO_CLIP_MIN, ACCEL_IO_CLIP_MAX);

			accel_io_t out_y = acc;

			// Activation Function
			out_y = (apply_relu && (out_y < 0)) ? ((accel_io_t) 0) : out_y;

			y[IDX2D(w_d1, x_d1, x_dim_1)] = out_y;
		}
	}
}


// Here the 1st dimension of x and (w, b, s) are affined
template<uint32_t x_dim_0, uint32_t x_dim_1>
void gold_affine_transpose(
	const accel_io_t *x,
	const accel_io_t *res,
	const accel_w_t *w,
	const accel_b_t *b,
	const accel_s_t *s,
	const bool add_bias,
	const bool transpose_input,
	const bool transpose_output,
	const bool add_residual,
	accel_io_t *y
	) {

	for (uint32_t x_d1 = 0; x_d1 < x_dim_1; x_d1++) {
		for (uint32_t x_d0 = 0; x_d0 < x_dim_0; x_d0++) {

			accel_io_t x_in;
			if (transpose_input)
				x_in = x[IDX2D(x_d1, x_d0, x_dim_0)];
			else
				x_in = x[IDX2D(x_d0, x_d1, x_dim_1)];

			accel_mul_t acc = x_in * w[x_d0];

			ROUNDED_SHIFT(acc, s[x_d0]);
			acc += (add_bias) ? b[x_d0] : (accel_b_t) 0;

			// Residual
			acc += (add_residual) ? res[IDX2D(x_d0, x_d1, x_dim_1)] : ((accel_io_t) 0);

			CLIP(acc, ACCEL_IO_CLIP_MIN, ACCEL_IO_CLIP_MAX);

			accel_io_t out_y = acc;

			if (transpose_output)
				y[IDX2D(x_d1, x_d0, x_dim_0)] = out_y;
			else
				y[IDX2D(x_d0, x_d1, x_dim_1)] = out_y;
		}
	}
}


void gold_resmlp(
	accel_io_t hardware_out_y[Y_DIM_0 * Y_DIM_1]
	) {

	accel_io_t buff_0[RM_EM_OUT_DIM_0 * RM_EM_OUT_DIM_1];
	accel_io_t res_buff_0[RM_CP_IN_DIM_1 * RM_CP_IN_DIM_0];
	accel_io_t buff_1[RM_CP_IN_DIM_1 * RM_CP_IN_DIM_0];
	accel_io_t buff_2[RM_CP_IN_DIM_1 * RM_CP_IN_DIM_0];

	gold_linear<RM_EM_IN_DIM_0, RM_EM_IN_DIM_1, RM_EM_L0_W_DIM_1>(
		TO_BUFF2(samples_input),
		NULL,
		TO_BUFF2(embed_linear_w),
		embed_linear_b,
		embed_linear_s,
		true,
		false,
		false,
		buff_0
	);

	gold_affine_transpose<RM_CP_IN_DIM_0, RM_CP_IN_DIM_1>(
		buff_0,
		NULL,
		blocks_0_token_mixer_affine_1_w,
		blocks_0_token_mixer_affine_1_b,
		blocks_0_token_mixer_affine_1_s,
		true,
		false,
		true,
		false,
		buff_1
	);

	gold_affine_transpose<RM_CP_IN_DIM_0, RM_CP_IN_DIM_1>(
		buff_0,
		NULL,
		blocks_0_token_mixer_res_affine_w,
		blocks_0_token_mixer_res_affine_b,
		blocks_0_token_mixer_res_affine_s,
		true,
		false,
		false,
		false,
		res_buff_0
	);

	gold_linear<RM_CP_FC_W_DIM_0, RM_CP_IN_DIM_0, RM_CP_FC_W_DIM_1>(
		buff_1,
		NULL,
		TO_BUFF2(blocks_0_token_mixer_linear_w),
		blocks_0_token_mixer_linear_b,
		blocks_0_token_mixer_linear_s,
		true,
		false,
		false,
		buff_2
	);

	gold_affine_transpose<RM_CP_IN_DIM_0, RM_CP_IN_DIM_1>(
		buff_2,
		res_buff_0,
		blocks_0_token_mixer_affine_2_w,
		blocks_0_token_mixer_affine_2_b,
		blocks_0_token_mixer_affine_2_s,
		true,
		true,
		false,
		true,
		hardware_out_y
	);
}


int main () {

	// accel_io_t hardware_out_y[Y_DIM_0 * Y_DIM_1];
	// gold_resmlp(hardware_out_y);

	// ================================================================================
	// Hardware Kernel
	// ================================================================================

	std::cout << "Allocating and Setting Hardware Buffers" << std::endl;

	double hardware_in_x_size =
		sizeof(accel_io_vec_t) * ceil((RESMLP_IN_DIM_0 * RESMLP_IN_DIM_1) / (double) PORT_VEC_SIZE);
	size_t hardware_in_x_size_bytes = hardware_in_x_size;
	std:: cout << sizeof(accel_io_vec_t) * (RESMLP_IN_DIM_0 * RESMLP_IN_DIM_1) / PORT_VEC_SIZE
		<< " " << hardware_in_x_size_bytes << "\n";

	accel_io_vec_t *hardware_in_x = NULL;
	posix_memalign((void**)&hardware_in_x, 4096, hardware_in_x_size_bytes);

	copy_to_vec<accel_io_t, accel_io_vec_t, PORT_VEC_SIZE>(
		CAST_TO_1D_ARRAY(samples_input, accel_io_t),
		hardware_in_x,
		RESMLP_IN_DIM_0 * RESMLP_IN_DIM_1
	);

	// ---------------------------------------------------------------------

	double hardware_out_y_size =
		sizeof(accel_io_vec_t) * ceil((RESMLP_OUT_DIM_0 * RESMLP_OUT_DIM_1) / (double) PORT_VEC_SIZE);
	size_t hardware_out_y_size_bytes = hardware_out_y_size;
	std:: cout << sizeof(accel_io_vec_t) * (RESMLP_OUT_DIM_0 * RESMLP_OUT_DIM_1) / PORT_VEC_SIZE
		<< " " << hardware_out_y_size_bytes << "\n";

	accel_io_vec_t *hardware_out_y = NULL;
	posix_memalign((void**)&hardware_out_y, 4096, hardware_out_y_size_bytes);

	set_vecs<accel_io_vec_t, accel_io_t, PORT_VEC_SIZE>(
		hardware_out_y,
		0,
		RESMLP_OUT_DIM_0 * RESMLP_OUT_DIM_1
	);

	// ---------------------------------------------------------------------
	// MARK: Weights
	// ---------------------------------------------------------------------

	std::cout << "Weights..." << std::endl;

	size_t hardware_w_size_bytes = 0;

	// Store the indices from which the layer weights will be stored
	hardware_w_size_bytes += (RM_EM_L0_W_DIM_0 * RM_EM_L0_W_DIM_1);
	// Layer 1
	hardware_w_size_bytes += (RM_CP_FC_W_DIM_0 * RM_CP_FC_W_DIM_1);
	hardware_w_size_bytes += (RM_CC_FC1_W_DIM_0 * RM_CC_FC1_W_DIM_1);
	hardware_w_size_bytes += (RM_CC_FC2_W_DIM_0 * RM_CC_FC2_W_DIM_1);
	// Layer 2
	hardware_w_size_bytes += (RM_CP_FC_W_DIM_0 * RM_CP_FC_W_DIM_1);
	hardware_w_size_bytes += (RM_CC_FC1_W_DIM_0 * RM_CC_FC1_W_DIM_1);
	hardware_w_size_bytes += (RM_CC_FC2_W_DIM_0 * RM_CC_FC2_W_DIM_1);
	// Layer 3
	hardware_w_size_bytes += (RM_CP_FC_W_DIM_0 * RM_CP_FC_W_DIM_1);
	hardware_w_size_bytes += (RM_CC_FC1_W_DIM_0 * RM_CC_FC1_W_DIM_1);
	hardware_w_size_bytes += (RM_CC_FC2_W_DIM_0 * RM_CC_FC2_W_DIM_1);
	// Layer 4
	hardware_w_size_bytes += (RM_CP_FC_W_DIM_0 * RM_CP_FC_W_DIM_1);
	hardware_w_size_bytes += (RM_CC_FC1_W_DIM_0 * RM_CC_FC1_W_DIM_1);
	hardware_w_size_bytes += (RM_CC_FC2_W_DIM_0 * RM_CC_FC2_W_DIM_1);
	// Output layer
	hardware_w_size_bytes += (RM_OUTP_FC_W_DIM_0 * CEIL_MULTIPLE(RM_OUTP_FC_W_DIM_1, PORT_VEC_SIZE));

	// Apply Ceiling when mapping to Port Vector Size
	hardware_w_size_bytes = ceil(hardware_w_size_bytes / (double) PORT_VEC_SIZE);
	hardware_w_size_bytes *= sizeof(accel_w_vec_t);

	accel_w_vec_t *hardware_w = NULL;
	std::cout << hardware_w_size_bytes << std::endl;
	posix_memalign((void**)&hardware_w, 4096, hardware_w_size_bytes);

	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		CAST_TO_1D_ARRAY(embed_linear_w, accel_w_t),
		hardware_w + (RM_EM_L0_W_OFFSET / PORT_VEC_SIZE),
		RM_EM_L0_W_DIM_0,
		RM_EM_L0_W_DIM_1
	);
	// Layer 1
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		CAST_TO_1D_ARRAY(blocks_0_token_mixer_linear_w, accel_w_t),
		hardware_w + ((RM_LAYER_1_FC_W_OFFSET+RM_CP_FC_W_OFFSET) / PORT_VEC_SIZE),
		RM_CP_FC_W_DIM_0,
		RM_CP_FC_W_DIM_1
	);
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		CAST_TO_1D_ARRAY(blocks_0_mlp_linear_1_w, accel_w_t),
		hardware_w + ((RM_LAYER_1_FC_W_OFFSET+RM_CC_FC1_W_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC1_W_DIM_0,
		RM_CC_FC1_W_DIM_1
	);
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		CAST_TO_1D_ARRAY(blocks_0_mlp_linear_2_w, accel_w_t),
		hardware_w + ((RM_LAYER_1_FC_W_OFFSET+RM_CC_FC2_W_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC2_W_DIM_0,
		RM_CC_FC2_W_DIM_1
	);
	// Layer 2
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		CAST_TO_1D_ARRAY(blocks_1_token_mixer_linear_w, accel_w_t),
		hardware_w + ((RM_LAYER_2_FC_W_OFFSET+RM_CP_FC_W_OFFSET) / PORT_VEC_SIZE),
		RM_CP_FC_W_DIM_0,
		RM_CP_FC_W_DIM_1
	);
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		CAST_TO_1D_ARRAY(blocks_1_mlp_linear_1_w, accel_w_t),
		hardware_w + ((RM_LAYER_2_FC_W_OFFSET+RM_CC_FC1_W_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC1_W_DIM_0,
		RM_CC_FC1_W_DIM_1
	);
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		CAST_TO_1D_ARRAY(blocks_1_mlp_linear_2_w, accel_w_t),
		hardware_w + ((RM_LAYER_2_FC_W_OFFSET+RM_CC_FC2_W_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC2_W_DIM_0,
		RM_CC_FC2_W_DIM_1
	);
	// Layer 3
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		CAST_TO_1D_ARRAY(blocks_2_token_mixer_linear_w, accel_w_t),
		hardware_w + ((RM_LAYER_3_FC_W_OFFSET+RM_CP_FC_W_OFFSET) / PORT_VEC_SIZE),
		RM_CP_FC_W_DIM_0,
		RM_CP_FC_W_DIM_1
	);
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		CAST_TO_1D_ARRAY(blocks_2_mlp_linear_1_w, accel_w_t),
		hardware_w + ((RM_LAYER_3_FC_W_OFFSET+RM_CC_FC1_W_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC1_W_DIM_0,
		RM_CC_FC1_W_DIM_1
	);
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		CAST_TO_1D_ARRAY(blocks_2_mlp_linear_2_w, accel_w_t),
		hardware_w + ((RM_LAYER_3_FC_W_OFFSET+RM_CC_FC2_W_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC2_W_DIM_0,
		RM_CC_FC2_W_DIM_1
	);
	// Layer 4
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		CAST_TO_1D_ARRAY(blocks_3_token_mixer_linear_w, accel_w_t),
		hardware_w + ((RM_LAYER_4_FC_W_OFFSET+RM_CP_FC_W_OFFSET) / PORT_VEC_SIZE),
		RM_CP_FC_W_DIM_0,
		RM_CP_FC_W_DIM_1
	);
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		CAST_TO_1D_ARRAY(blocks_3_mlp_linear_1_w, accel_w_t),
		hardware_w + ((RM_LAYER_4_FC_W_OFFSET+RM_CC_FC1_W_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC1_W_DIM_0,
		RM_CC_FC1_W_DIM_1
	);
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		CAST_TO_1D_ARRAY(blocks_3_mlp_linear_2_w, accel_w_t),
		hardware_w + ((RM_LAYER_4_FC_W_OFFSET+RM_CC_FC2_W_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC2_W_DIM_0,
		RM_CC_FC2_W_DIM_1
	);
	// Output layer
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		CAST_TO_1D_ARRAY(classifier_linear_w, accel_w_t),
		hardware_w + ((RM_O_LAYER_FC_W_OFFSET+RM_OUTP_FC_W_OFFSET) / PORT_VEC_SIZE),
		RM_OUTP_FC_W_DIM_0,
		RM_OUTP_FC_W_DIM_1
	);

	// ---------------------------------------------------------------------
	// MARK: Affine Weights
	// ---------------------------------------------------------------------

	std::cout << "Affine Weights..." << std::endl;

	size_t hardware_w_aff_size_bytes = 0;

	// Store the indices from which the layer weights will be stored
	// Layer 1
	hardware_w_aff_size_bytes += (RM_CP_RES_AFF_W_DIM_0);
	hardware_w_aff_size_bytes += (RM_CP_AFF_1_W_DIM_0);
	hardware_w_aff_size_bytes += (RM_CP_AFF_2_W_DIM_0);
	hardware_w_aff_size_bytes += (RM_CC_RES_AFF_W_DIM_0);
	// Layer 2
	hardware_w_aff_size_bytes += (RM_CP_RES_AFF_W_DIM_0);
	hardware_w_aff_size_bytes += (RM_CP_AFF_1_W_DIM_0);
	hardware_w_aff_size_bytes += (RM_CP_AFF_2_W_DIM_0);
	hardware_w_aff_size_bytes += (RM_CC_RES_AFF_W_DIM_0);
	// Layer 3
	hardware_w_aff_size_bytes += (RM_CP_RES_AFF_W_DIM_0);
	hardware_w_aff_size_bytes += (RM_CP_AFF_1_W_DIM_0);
	hardware_w_aff_size_bytes += (RM_CP_AFF_2_W_DIM_0);
	hardware_w_aff_size_bytes += (RM_CC_RES_AFF_W_DIM_0);
	// Layer 4
	hardware_w_aff_size_bytes += (RM_CP_RES_AFF_W_DIM_0);
	hardware_w_aff_size_bytes += (RM_CP_AFF_1_W_DIM_0);
	hardware_w_aff_size_bytes += (RM_CP_AFF_2_W_DIM_0);
	hardware_w_aff_size_bytes += (RM_CC_RES_AFF_W_DIM_0);

	// Apply Ceiling when mapping to Port Vector Size
	hardware_w_aff_size_bytes = ceil(hardware_w_aff_size_bytes / (double) PORT_VEC_SIZE);
	hardware_w_aff_size_bytes *= sizeof(accel_w_vec_t);

	accel_w_vec_t *hardware_w_aff = NULL;
	std::cout << hardware_w_aff_size_bytes << std::endl;
	posix_memalign((void**)&hardware_w_aff, 4096, hardware_w_aff_size_bytes);

	// Layer 1
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		blocks_0_token_mixer_affine_1_w,
		hardware_w_aff + ((RM_LAYER_1_AFF_W_OFFSET+RM_CP_AFF_1_W_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_1_W_DIM_0
	);
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		blocks_0_token_mixer_affine_2_w,
		hardware_w_aff + ((RM_LAYER_1_AFF_W_OFFSET+RM_CP_AFF_2_W_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_2_W_DIM_0
	);
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		blocks_0_token_mixer_res_affine_w,
		hardware_w_aff + ((RM_LAYER_1_AFF_W_OFFSET+RM_CP_RES_AFF_W_OFFSET) / PORT_VEC_SIZE),
		RM_CP_RES_AFF_W_DIM_0
	);
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		blocks_0_mlp_res_affine_w,
		hardware_w_aff + ((RM_LAYER_1_AFF_W_OFFSET+RM_CC_RES_AFF_W_OFFSET) / PORT_VEC_SIZE),
		RM_CC_RES_AFF_W_DIM_0
	);
	// Layer 2
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		blocks_1_token_mixer_affine_1_w,
		hardware_w_aff + ((RM_LAYER_2_AFF_W_OFFSET+RM_CP_AFF_1_W_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_1_W_DIM_0
	);
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		blocks_1_token_mixer_affine_2_w,
		hardware_w_aff + ((RM_LAYER_2_AFF_W_OFFSET+RM_CP_AFF_2_W_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_2_W_DIM_0
	);
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		blocks_1_token_mixer_res_affine_w,
		hardware_w_aff + ((RM_LAYER_2_AFF_W_OFFSET+RM_CP_RES_AFF_W_OFFSET) / PORT_VEC_SIZE),
		RM_CP_RES_AFF_W_DIM_0
	);
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		blocks_1_mlp_res_affine_w,
		hardware_w_aff + ((RM_LAYER_2_AFF_W_OFFSET+RM_CC_RES_AFF_W_OFFSET) / PORT_VEC_SIZE),
		RM_CC_RES_AFF_W_DIM_0
	);
	// Layer 3
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		blocks_2_token_mixer_affine_1_w,
		hardware_w_aff + ((RM_LAYER_3_AFF_W_OFFSET+RM_CP_AFF_1_W_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_1_W_DIM_0
	);
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		blocks_2_token_mixer_affine_2_w,
		hardware_w_aff + ((RM_LAYER_3_AFF_W_OFFSET+RM_CP_AFF_2_W_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_2_W_DIM_0
	);
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		blocks_2_token_mixer_res_affine_w,
		hardware_w_aff + ((RM_LAYER_3_AFF_W_OFFSET+RM_CP_RES_AFF_W_OFFSET) / PORT_VEC_SIZE),
		RM_CP_RES_AFF_W_DIM_0
	);
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		blocks_2_mlp_res_affine_w,
		hardware_w_aff + ((RM_LAYER_3_AFF_W_OFFSET+RM_CC_RES_AFF_W_OFFSET) / PORT_VEC_SIZE),
		RM_CC_RES_AFF_W_DIM_0
	);
	// Layer 4
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		blocks_3_token_mixer_affine_1_w,
		hardware_w_aff + ((RM_LAYER_4_AFF_W_OFFSET+RM_CP_AFF_1_W_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_1_W_DIM_0
	);
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		blocks_3_token_mixer_affine_2_w,
		hardware_w_aff + ((RM_LAYER_4_AFF_W_OFFSET+RM_CP_AFF_2_W_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_2_W_DIM_0
	);
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		blocks_3_token_mixer_res_affine_w,
		hardware_w_aff + ((RM_LAYER_4_AFF_W_OFFSET+RM_CP_RES_AFF_W_OFFSET) / PORT_VEC_SIZE),
		RM_CP_RES_AFF_W_DIM_0
	);
	copy_to_vec<accel_w_t, accel_w_vec_t, PORT_VEC_SIZE>(
		blocks_3_mlp_res_affine_w,
		hardware_w_aff + ((RM_LAYER_4_AFF_W_OFFSET+RM_CC_RES_AFF_W_OFFSET) / PORT_VEC_SIZE),
		RM_CC_RES_AFF_W_DIM_0
	);

	// ---------------------------------------------------------------------
	// MARK: Biases
	// ---------------------------------------------------------------------

	std::cout << "Biases..." << std::endl;

	size_t hardware_b_size_bytes = 0;

	// Store the indices from which the layer weights will be stored
	hardware_b_size_bytes += (RM_EM_L0_W_DIM_1);
	// Layer 1
	hardware_b_size_bytes += (RM_CP_FC_W_DIM_1);
	hardware_b_size_bytes += (RM_CC_FC1_W_DIM_1);
	hardware_b_size_bytes += (RM_CC_FC2_W_DIM_1);
	// Layer 2
	hardware_b_size_bytes += (RM_CP_FC_W_DIM_1);
	hardware_b_size_bytes += (RM_CC_FC1_W_DIM_1);
	hardware_b_size_bytes += (RM_CC_FC2_W_DIM_1);
	// Layer 1
	hardware_b_size_bytes += (RM_CP_FC_W_DIM_1);
	hardware_b_size_bytes += (RM_CC_FC1_W_DIM_1);
	hardware_b_size_bytes += (RM_CC_FC2_W_DIM_1);
	// Layer 2
	hardware_b_size_bytes += (RM_CP_FC_W_DIM_1);
	hardware_b_size_bytes += (RM_CC_FC1_W_DIM_1);
	hardware_b_size_bytes += (RM_CC_FC2_W_DIM_1);
	// Output layer
	hardware_b_size_bytes += (RM_OUTP_FC_W_DIM_1);

	// Apply Ceiling when mapping to Port Vector Size
	hardware_b_size_bytes = ceil(hardware_b_size_bytes / (double) PORT_VEC_SIZE);
	hardware_b_size_bytes *= sizeof(accel_b_vec_t);

	accel_b_vec_t *hardware_b = NULL;
	std::cout << hardware_b_size_bytes << std::endl;
	posix_memalign((void**)&hardware_b, 4096, hardware_b_size_bytes);

	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		embed_linear_b,
		hardware_b + ((RM_EM_L0_BS_OFFSET) / PORT_VEC_SIZE),
		RM_EM_L0_W_DIM_1
	);
	// Layer 1
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_0_token_mixer_linear_b,
		hardware_b + ((RM_LAYER_1_FC_BS_OFFSET+RM_CP_FC_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_FC_W_DIM_1
	);
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_0_mlp_linear_1_b,
		hardware_b + ((RM_LAYER_1_FC_BS_OFFSET+RM_CC_FC1_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC1_W_DIM_1
	);
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_0_mlp_linear_2_b,
		hardware_b + ((RM_LAYER_1_FC_BS_OFFSET+RM_CC_FC2_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC2_W_DIM_1
	);
	// Layer 2
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_1_token_mixer_linear_b,
		hardware_b + ((RM_LAYER_2_FC_BS_OFFSET+RM_CP_FC_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_FC_W_DIM_1
	);
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_1_mlp_linear_1_b,
		hardware_b + ((RM_LAYER_2_FC_BS_OFFSET+RM_CC_FC1_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC1_W_DIM_1
	);
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_1_mlp_linear_2_b,
		hardware_b + ((RM_LAYER_2_FC_BS_OFFSET+RM_CC_FC2_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC2_W_DIM_1
	);
	// Layer 3
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_2_token_mixer_linear_b,
		hardware_b + ((RM_LAYER_3_FC_BS_OFFSET+RM_CP_FC_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_FC_W_DIM_1
	);
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_2_mlp_linear_1_b,
		hardware_b + ((RM_LAYER_3_FC_BS_OFFSET+RM_CC_FC1_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC1_W_DIM_1
	);
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_2_mlp_linear_2_b,
		hardware_b + ((RM_LAYER_3_FC_BS_OFFSET+RM_CC_FC2_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC2_W_DIM_1
	);
	// Layer 4
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_3_token_mixer_linear_b,
		hardware_b + ((RM_LAYER_4_FC_BS_OFFSET+RM_CP_FC_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_FC_W_DIM_1
	);
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_3_mlp_linear_1_b,
		hardware_b + ((RM_LAYER_4_FC_BS_OFFSET+RM_CC_FC1_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC1_W_DIM_1
	);
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_3_mlp_linear_2_b,
		hardware_b + ((RM_LAYER_4_FC_BS_OFFSET+RM_CC_FC2_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC2_W_DIM_1
	);
	// Output layer
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		classifier_linear_b,
		hardware_b + ((RM_O_LAYER_FC_BS_OFFSET+RM_OUTP_FC_BS_OFFSET) / PORT_VEC_SIZE),
		RM_OUTP_FC_W_DIM_1
	);

	// ---------------------------------------------------------------------
	// MARK: Affine Biases
	// ---------------------------------------------------------------------

	std::cout << "Affine Biases..." << std::endl;

	size_t hardware_b_aff_size_bytes = 0;

	// Store the indices from which the layer weights will be stored
	// Layer 1
	hardware_b_aff_size_bytes += (RM_CP_RES_AFF_W_DIM_0);
	hardware_b_aff_size_bytes += (RM_CP_AFF_1_W_DIM_0);
	hardware_b_aff_size_bytes += (RM_CP_AFF_2_W_DIM_0);
	hardware_b_aff_size_bytes += (RM_CC_RES_AFF_W_DIM_0);
	// Layer 2
	hardware_b_aff_size_bytes += (RM_CP_RES_AFF_W_DIM_0);
	hardware_b_aff_size_bytes += (RM_CP_AFF_1_W_DIM_0);
	hardware_b_aff_size_bytes += (RM_CP_AFF_2_W_DIM_0);
	hardware_b_aff_size_bytes += (RM_CC_RES_AFF_W_DIM_0);
	// Layer 3
	hardware_b_aff_size_bytes += (RM_CP_RES_AFF_W_DIM_0);
	hardware_b_aff_size_bytes += (RM_CP_AFF_1_W_DIM_0);
	hardware_b_aff_size_bytes += (RM_CP_AFF_2_W_DIM_0);
	hardware_b_aff_size_bytes += (RM_CC_RES_AFF_W_DIM_0);
	// Layer 4
	hardware_b_aff_size_bytes += (RM_CP_RES_AFF_W_DIM_0);
	hardware_b_aff_size_bytes += (RM_CP_AFF_1_W_DIM_0);
	hardware_b_aff_size_bytes += (RM_CP_AFF_2_W_DIM_0);
	hardware_b_aff_size_bytes += (RM_CC_RES_AFF_W_DIM_0);

	// Apply Ceiling when mapping to Port Vector Size
	hardware_b_aff_size_bytes = ceil(hardware_b_aff_size_bytes / (double) PORT_VEC_SIZE);
	hardware_b_aff_size_bytes *= sizeof(accel_b_vec_t);

	accel_b_vec_t *hardware_b_aff = NULL;
	std::cout << hardware_b_aff_size_bytes << std::endl;
	posix_memalign((void**)&hardware_b_aff, 4096, hardware_b_aff_size_bytes);

	// Layer 1
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_0_token_mixer_affine_1_b,
		hardware_b_aff + ((RM_LAYER_1_AFF_BS_OFFSET+RM_CP_AFF_1_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_1_W_DIM_0
	);
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_0_token_mixer_affine_2_b,
		hardware_b_aff + ((RM_LAYER_1_AFF_BS_OFFSET+RM_CP_AFF_2_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_2_W_DIM_0
	);
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_0_token_mixer_res_affine_b,
		hardware_b_aff + ((RM_LAYER_1_AFF_BS_OFFSET+RM_CP_RES_AFF_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_RES_AFF_W_DIM_0
	);
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_0_mlp_res_affine_b,
		hardware_b_aff + ((RM_LAYER_1_AFF_BS_OFFSET+RM_CC_RES_AFF_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_RES_AFF_W_DIM_0
	);
	// Layer 2
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_1_token_mixer_affine_1_b,
		hardware_b_aff + ((RM_LAYER_2_AFF_BS_OFFSET+RM_CP_AFF_1_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_1_W_DIM_0
	);
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_1_token_mixer_affine_2_b,
		hardware_b_aff + ((RM_LAYER_2_AFF_BS_OFFSET+RM_CP_AFF_2_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_2_W_DIM_0
	);
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_1_token_mixer_res_affine_b,
		hardware_b_aff + ((RM_LAYER_2_AFF_BS_OFFSET+RM_CP_RES_AFF_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_RES_AFF_W_DIM_0
	);
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_1_mlp_res_affine_b,
		hardware_b_aff + ((RM_LAYER_2_AFF_BS_OFFSET+RM_CC_RES_AFF_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_RES_AFF_W_DIM_0
	);
	// Layer 3
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_2_token_mixer_affine_1_b,
		hardware_b_aff + ((RM_LAYER_3_AFF_BS_OFFSET+RM_CP_AFF_1_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_1_W_DIM_0
	);
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_2_token_mixer_affine_2_b,
		hardware_b_aff + ((RM_LAYER_3_AFF_BS_OFFSET+RM_CP_AFF_2_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_2_W_DIM_0
	);
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_2_token_mixer_res_affine_b,
		hardware_b_aff + ((RM_LAYER_3_AFF_BS_OFFSET+RM_CP_RES_AFF_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_RES_AFF_W_DIM_0
	);
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_2_mlp_res_affine_b,
		hardware_b_aff + ((RM_LAYER_3_AFF_BS_OFFSET+RM_CC_RES_AFF_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_RES_AFF_W_DIM_0
	);
	// Layer 4
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_3_token_mixer_affine_1_b,
		hardware_b_aff + ((RM_LAYER_4_AFF_BS_OFFSET+RM_CP_AFF_1_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_1_W_DIM_0
	);
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_3_token_mixer_affine_2_b,
		hardware_b_aff + ((RM_LAYER_4_AFF_BS_OFFSET+RM_CP_AFF_2_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_2_W_DIM_0
	);
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_3_token_mixer_res_affine_b,
		hardware_b_aff + ((RM_LAYER_4_AFF_BS_OFFSET+RM_CP_RES_AFF_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_RES_AFF_W_DIM_0
	);
	copy_to_vec<accel_b_t, accel_b_vec_t, PORT_VEC_SIZE>(
		blocks_3_mlp_res_affine_b,
		hardware_b_aff + ((RM_LAYER_4_AFF_BS_OFFSET+RM_CC_RES_AFF_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_RES_AFF_W_DIM_0
	);

	// ---------------------------------------------------------------------
	// MARK: Scalars
	// ---------------------------------------------------------------------

	std::cout << "Scalars..." << std::endl;

	size_t hardware_s_size_bytes = 0;

	// Store the indices from which the layer weights will be stored
	hardware_s_size_bytes += (RM_EM_L0_W_DIM_1);
	// Layer 1
	hardware_s_size_bytes += (RM_CP_FC_W_DIM_1);
	hardware_s_size_bytes += (RM_CC_FC1_W_DIM_1);
	hardware_s_size_bytes += (RM_CC_FC2_W_DIM_1);
	// Layer 2
	hardware_s_size_bytes += (RM_CP_FC_W_DIM_1);
	hardware_s_size_bytes += (RM_CC_FC1_W_DIM_1);
	hardware_s_size_bytes += (RM_CC_FC2_W_DIM_1);
	// Layer 3
	hardware_s_size_bytes += (RM_CP_FC_W_DIM_1);
	hardware_s_size_bytes += (RM_CC_FC1_W_DIM_1);
	hardware_s_size_bytes += (RM_CC_FC2_W_DIM_1);
	// Layer 4
	hardware_s_size_bytes += (RM_CP_FC_W_DIM_1);
	hardware_s_size_bytes += (RM_CC_FC1_W_DIM_1);
	hardware_s_size_bytes += (RM_CC_FC2_W_DIM_1);
	// Output layer
	hardware_s_size_bytes += (RM_OUTP_FC_W_DIM_1);

	// Apply Ceiling when mapping to Port Vector Size
	hardware_s_size_bytes = ceil(hardware_s_size_bytes / (double) PORT_VEC_SIZE);
	hardware_s_size_bytes *= sizeof(accel_s_vec_t);

	accel_s_vec_t *hardware_s = NULL;
	posix_memalign((void**)&hardware_s, 4096, hardware_s_size_bytes);
	std::cout << hardware_s_size_bytes << std::endl;

	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		embed_linear_s,
		hardware_s + ((RM_EM_L0_BS_OFFSET) / PORT_VEC_SIZE),
		RM_EM_L0_W_DIM_1
	);
	// Layer 1
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_0_token_mixer_linear_s,
		hardware_s + ((RM_LAYER_1_FC_BS_OFFSET+RM_CP_FC_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_FC_W_DIM_1
	);
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_0_mlp_linear_1_s,
		hardware_s + ((RM_LAYER_1_FC_BS_OFFSET+RM_CC_FC1_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC1_W_DIM_1
	);
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_0_mlp_linear_2_s,
		hardware_s + ((RM_LAYER_1_FC_BS_OFFSET+RM_CC_FC2_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC2_W_DIM_1
	);
	// Layer 2
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_1_token_mixer_linear_s,
		hardware_s + ((RM_LAYER_2_FC_BS_OFFSET+RM_CP_FC_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_FC_W_DIM_1
	);
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_1_mlp_linear_1_s,
		hardware_s + ((RM_LAYER_2_FC_BS_OFFSET+RM_CC_FC1_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC1_W_DIM_1
	);
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_1_mlp_linear_2_s,
		hardware_s + ((RM_LAYER_2_FC_BS_OFFSET+RM_CC_FC2_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC2_W_DIM_1
	);
	// Layer 3
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_2_token_mixer_linear_s,
		hardware_s + ((RM_LAYER_3_FC_BS_OFFSET+RM_CP_FC_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_FC_W_DIM_1
	);
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_2_mlp_linear_1_s,
		hardware_s + ((RM_LAYER_3_FC_BS_OFFSET+RM_CC_FC1_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC1_W_DIM_1
	);
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_2_mlp_linear_2_s,
		hardware_s + ((RM_LAYER_3_FC_BS_OFFSET+RM_CC_FC2_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC2_W_DIM_1
	);
	// Layer 4
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_3_token_mixer_linear_s,
		hardware_s + ((RM_LAYER_4_FC_BS_OFFSET+RM_CP_FC_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_FC_W_DIM_1
	);
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_3_mlp_linear_1_s,
		hardware_s + ((RM_LAYER_4_FC_BS_OFFSET+RM_CC_FC1_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC1_W_DIM_1
	);
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_3_mlp_linear_2_s,
		hardware_s + ((RM_LAYER_4_FC_BS_OFFSET+RM_CC_FC2_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_FC2_W_DIM_1
	);
	// Layer 4
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		classifier_linear_s,
		hardware_s + ((RM_O_LAYER_FC_BS_OFFSET+RM_OUTP_FC_BS_OFFSET) / PORT_VEC_SIZE),
		RM_OUTP_FC_W_DIM_1
	);

	// ---------------------------------------------------------------------
	// MARK: Affine Scalars
	// ---------------------------------------------------------------------

	std::cout << "Affine Scalars..." << std::endl;

	size_t hardware_s_aff_size_bytes = 0;

	// Store the indices from which the layer weights will be stored
	// Layer 1
	hardware_s_aff_size_bytes += (RM_CP_RES_AFF_W_DIM_0);
	hardware_s_aff_size_bytes += (RM_CP_AFF_1_W_DIM_0);
	hardware_s_aff_size_bytes += (RM_CP_AFF_2_W_DIM_0);
	hardware_s_aff_size_bytes += (RM_CC_RES_AFF_W_DIM_0);
	// Layer 2
	hardware_s_aff_size_bytes += (RM_CP_RES_AFF_W_DIM_0);
	hardware_s_aff_size_bytes += (RM_CP_AFF_1_W_DIM_0);
	hardware_s_aff_size_bytes += (RM_CP_AFF_2_W_DIM_0);
	hardware_s_aff_size_bytes += (RM_CC_RES_AFF_W_DIM_0);
	// Layer 1
	hardware_s_aff_size_bytes += (RM_CP_RES_AFF_W_DIM_0);
	hardware_s_aff_size_bytes += (RM_CP_AFF_1_W_DIM_0);
	hardware_s_aff_size_bytes += (RM_CP_AFF_2_W_DIM_0);
	hardware_s_aff_size_bytes += (RM_CC_RES_AFF_W_DIM_0);
	// Layer 2
	hardware_s_aff_size_bytes += (RM_CP_RES_AFF_W_DIM_0);
	hardware_s_aff_size_bytes += (RM_CP_AFF_1_W_DIM_0);
	hardware_s_aff_size_bytes += (RM_CP_AFF_2_W_DIM_0);
	hardware_s_aff_size_bytes += (RM_CC_RES_AFF_W_DIM_0);

	// Apply Ceiling when mapping to Port Vector Size
	hardware_s_aff_size_bytes = ceil(hardware_s_aff_size_bytes / (double) PORT_VEC_SIZE);
	hardware_s_aff_size_bytes *= sizeof(accel_s_vec_t);

	accel_s_vec_t *hardware_s_aff = NULL;
	posix_memalign((void**)&hardware_s_aff, 4096, hardware_s_aff_size_bytes);
	std::cout << hardware_s_aff_size_bytes << std::endl;

	// Layer 1
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_0_token_mixer_affine_1_s,
		hardware_s_aff + ((RM_LAYER_1_AFF_BS_OFFSET+RM_CP_AFF_1_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_1_W_DIM_0
	);
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_0_token_mixer_affine_2_s,
		hardware_s_aff + ((RM_LAYER_1_AFF_BS_OFFSET+RM_CP_AFF_2_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_2_W_DIM_0
	);
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_0_token_mixer_res_affine_s,
		hardware_s_aff + ((RM_LAYER_1_AFF_BS_OFFSET+RM_CP_RES_AFF_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_RES_AFF_W_DIM_0
	);
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_0_mlp_res_affine_s,
		hardware_s_aff + ((RM_LAYER_1_AFF_BS_OFFSET+RM_CC_RES_AFF_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_RES_AFF_W_DIM_0
	);
	// Layer 2
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_1_token_mixer_affine_1_s,
		hardware_s_aff + ((RM_LAYER_2_AFF_BS_OFFSET+RM_CP_AFF_1_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_1_W_DIM_0
	);
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_1_token_mixer_affine_2_s,
		hardware_s_aff + ((RM_LAYER_2_AFF_BS_OFFSET+RM_CP_AFF_2_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_2_W_DIM_0
	);
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_1_token_mixer_res_affine_s,
		hardware_s_aff + ((RM_LAYER_2_AFF_BS_OFFSET+RM_CP_RES_AFF_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_RES_AFF_W_DIM_0
	);
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_1_mlp_res_affine_s,
		hardware_s_aff + ((RM_LAYER_2_AFF_BS_OFFSET+RM_CC_RES_AFF_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_RES_AFF_W_DIM_0
	);
	// Layer 3
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_2_token_mixer_affine_1_s,
		hardware_s_aff + ((RM_LAYER_3_AFF_BS_OFFSET+RM_CP_AFF_1_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_1_W_DIM_0
	);
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_2_token_mixer_affine_2_s,
		hardware_s_aff + ((RM_LAYER_3_AFF_BS_OFFSET+RM_CP_AFF_2_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_2_W_DIM_0
	);
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_2_token_mixer_res_affine_s,
		hardware_s_aff + ((RM_LAYER_3_AFF_BS_OFFSET+RM_CP_RES_AFF_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_RES_AFF_W_DIM_0
	);
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_2_mlp_res_affine_s,
		hardware_s_aff + ((RM_LAYER_3_AFF_BS_OFFSET+RM_CC_RES_AFF_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_RES_AFF_W_DIM_0
	);
	// Layer 4
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_3_token_mixer_affine_1_s,
		hardware_s_aff + ((RM_LAYER_4_AFF_BS_OFFSET+RM_CP_AFF_1_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_1_W_DIM_0
	);
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_3_token_mixer_affine_2_s,
		hardware_s_aff + ((RM_LAYER_4_AFF_BS_OFFSET+RM_CP_AFF_2_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_AFF_2_W_DIM_0
	);
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_3_token_mixer_res_affine_s,
		hardware_s_aff + ((RM_LAYER_4_AFF_BS_OFFSET+RM_CP_RES_AFF_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CP_RES_AFF_W_DIM_0
	);
	copy_to_vec<accel_s_t, accel_s_vec_t, PORT_VEC_SIZE>(
		blocks_3_mlp_res_affine_s,
		hardware_s_aff + ((RM_LAYER_4_AFF_BS_OFFSET+RM_CC_RES_AFF_BS_OFFSET) / PORT_VEC_SIZE),
		RM_CC_RES_AFF_W_DIM_0
	);

	// ---------------------------------------------------------------------
	// MARK: PSA LUT
	// ---------------------------------------------------------------------

	std::cout << "PSA LUTs..." << std::endl;

	size_t hardware_psa_lut_size_bytes = 0;

	// Store the indices from which the layer weights will be stored
	// Layer 1
	hardware_psa_lut_size_bytes += (RM_CC_PSA1_LUT_DIM_0 * RM_CC_PSA1_LUT_DIM_1);
	hardware_psa_lut_size_bytes += (RM_CC_PSA2_LUT_DIM_0 * RM_CC_PSA2_LUT_DIM_1);
	// Layer 2
	hardware_psa_lut_size_bytes += (RM_CC_PSA1_LUT_DIM_0 * RM_CC_PSA1_LUT_DIM_1);
	hardware_psa_lut_size_bytes += (RM_CC_PSA2_LUT_DIM_0 * RM_CC_PSA2_LUT_DIM_1);
	// Layer 3
	hardware_psa_lut_size_bytes += (RM_CC_PSA1_LUT_DIM_0 * RM_CC_PSA1_LUT_DIM_1);
	hardware_psa_lut_size_bytes += (RM_CC_PSA2_LUT_DIM_0 * RM_CC_PSA2_LUT_DIM_1);
	// Layer 4
	hardware_psa_lut_size_bytes += (RM_CC_PSA1_LUT_DIM_0 * RM_CC_PSA1_LUT_DIM_1);
	hardware_psa_lut_size_bytes += (RM_CC_PSA2_LUT_DIM_0 * RM_CC_PSA2_LUT_DIM_1);

	// Apply Ceiling when mapping to Port Vector Size
	hardware_psa_lut_size_bytes = ceil(hardware_psa_lut_size_bytes / (double) PORT_VEC_SIZE);
	hardware_psa_lut_size_bytes *= sizeof(accel_psa_lut_vec_t);

	accel_psa_lut_vec_t *hardware_psa_lut = NULL;
	std::cout << hardware_psa_lut_size_bytes << std::endl;
	posix_memalign((void**)&hardware_psa_lut, 4096, hardware_psa_lut_size_bytes);

	// Layer 1
	copy_to_vec<model_psa_t, accel_psa_lut_vec_t, PORT_VEC_SIZE>(
		CAST_TO_1D_ARRAY(blocks_0_mlp_psa_1_lerp_lut, model_psa_t),
		hardware_psa_lut + ((RM_LAYER_1_PSA_LUT_OFFSET+RM_CC_PSA1_LUT_OFFSET) / PORT_VEC_SIZE),
		RM_CC_PSA1_LUT_DIM_0,
		RM_CC_PSA1_LUT_DIM_1
	);
	copy_to_vec<model_psa_t, accel_psa_lut_vec_t, PORT_VEC_SIZE>(
		CAST_TO_1D_ARRAY(blocks_0_mlp_psa_2_lerp_lut, model_psa_t),
		hardware_psa_lut + ((RM_LAYER_1_PSA_LUT_OFFSET+RM_CC_PSA2_LUT_OFFSET) / PORT_VEC_SIZE),
		RM_CC_PSA2_LUT_DIM_0,
		RM_CC_PSA2_LUT_DIM_1
	);
	// Layer 2
	copy_to_vec<model_psa_t, accel_psa_lut_vec_t, PORT_VEC_SIZE>(
		CAST_TO_1D_ARRAY(blocks_1_mlp_psa_1_lerp_lut, model_psa_t),
		hardware_psa_lut + ((RM_LAYER_2_PSA_LUT_OFFSET+RM_CC_PSA1_LUT_OFFSET) / PORT_VEC_SIZE),
		RM_CC_PSA1_LUT_DIM_0,
		RM_CC_PSA1_LUT_DIM_1
	);
	copy_to_vec<model_psa_t, accel_psa_lut_vec_t, PORT_VEC_SIZE>(
		CAST_TO_1D_ARRAY(blocks_1_mlp_psa_2_lerp_lut, model_psa_t),
		hardware_psa_lut + ((RM_LAYER_2_PSA_LUT_OFFSET+RM_CC_PSA2_LUT_OFFSET) / PORT_VEC_SIZE),
		RM_CC_PSA2_LUT_DIM_0,
		RM_CC_PSA2_LUT_DIM_1
	);
	// Layer 3
	copy_to_vec<model_psa_t, accel_psa_lut_vec_t, PORT_VEC_SIZE>(
		CAST_TO_1D_ARRAY(blocks_2_mlp_psa_1_lerp_lut, model_psa_t),
		hardware_psa_lut + ((RM_LAYER_3_PSA_LUT_OFFSET+RM_CC_PSA1_LUT_OFFSET) / PORT_VEC_SIZE),
		RM_CC_PSA1_LUT_DIM_0,
		RM_CC_PSA1_LUT_DIM_1
	);
	copy_to_vec<model_psa_t, accel_psa_lut_vec_t, PORT_VEC_SIZE>(
		CAST_TO_1D_ARRAY(blocks_2_mlp_psa_2_lerp_lut, model_psa_t),
		hardware_psa_lut + ((RM_LAYER_3_PSA_LUT_OFFSET+RM_CC_PSA2_LUT_OFFSET) / PORT_VEC_SIZE),
		RM_CC_PSA2_LUT_DIM_0,
		RM_CC_PSA2_LUT_DIM_1
	);
	// Layer 4
	copy_to_vec<model_psa_t, accel_psa_lut_vec_t, PORT_VEC_SIZE>(
		CAST_TO_1D_ARRAY(blocks_3_mlp_psa_1_lerp_lut, model_psa_t),
		hardware_psa_lut + ((RM_LAYER_4_PSA_LUT_OFFSET+RM_CC_PSA1_LUT_OFFSET) / PORT_VEC_SIZE),
		RM_CC_PSA1_LUT_DIM_0,
		RM_CC_PSA1_LUT_DIM_1
	);
	copy_to_vec<model_psa_t, accel_psa_lut_vec_t, PORT_VEC_SIZE>(
		CAST_TO_1D_ARRAY(blocks_3_mlp_psa_2_lerp_lut, model_psa_t),
		hardware_psa_lut + ((RM_LAYER_4_PSA_LUT_OFFSET+RM_CC_PSA2_LUT_OFFSET) / PORT_VEC_SIZE),
		RM_CC_PSA2_LUT_DIM_0,
		RM_CC_PSA2_LUT_DIM_1
	);


	// for (int i = 0; i < 96; i++) {
	// 	for (int j = 0; j < 128; j++) {
	// 		int idx = (i * 128) + j;
	// 		size_t offset = ((RM_LAYER_2_PSA_LUT_OFFSET+RM_CC_PSA1_LUT_OFFSET) / PORT_VEC_SIZE);
	// 		std::cout << (hardware_psa_lut + offset)[idx / TILE_SIZE][idx % TILE_SIZE] << " ";
	// 	}
	// 	std::cout << std::endl;
	// }
	// exit(1);

	// ---------------------------------------------------------------------

	std::cout << "Calling kernel..." << std::endl;

	krnl_resmlp(
		hardware_in_x,
		hardware_w,
		hardware_b,
		hardware_s,
		hardware_w_aff,
		hardware_b_aff,
		hardware_s_aff,
		hardware_psa_lut,
		hardware_out_y
	);

	std::cout << "Done." << std::endl;

	// ================================================================================

	float err_sum = 0;
	float abs_err_sum = 0;
	int max_err_idx = -1;
	float max_abs_err = 0;
	for (int i = 0; i < Y_DIM_0; i++) {
		for (int j = 0; j < Y_DIM_1; j++) {
			int idx = IDX2D(i,j,Y_DIM_1);
			float err = ((float) REFERENCE_OUTPUT(i,j)) -
				(float) HARDWARE_OUTPUT(hardware_out_y, idx, i, j);
				// [idx / PORT_VEC_SIZE][idx % PORT_VEC_SIZE]);
				// (float)hardware_out_y[IDX2D(i,j,Y_DIM_1)]);
				// (float)hardware_out_y[i][j]);
			float abs_err = std::fabs(err);
			err_sum += err;
			abs_err_sum += abs_err;
			if (abs_err > max_abs_err) {
				max_abs_err = abs_err;
				max_err_idx = i;
			}
			err_sum += err;
		}
	}

//  max_err_idx = 0;
	if (max_err_idx == -1) {
		std::cout << "\nPERFECT MATCH" << std::endl;
	} else{
		std::cout << "\nAvg Err: " << err_sum / (Y_DIM_0 * Y_DIM_1) << std::endl;
		std::cout << "Avg Abs Err: " << abs_err_sum / (Y_DIM_0 * Y_DIM_1) << std::endl;
		std::cout << "Max Abs Err: " << max_abs_err << std::endl;
		std::cout << "HARDWARE\t\tREFERENCE\t" << "idx: " << max_err_idx << std::endl;
		for (int j = 0; j < Y_DIM_1; j++) {
			int idx = IDX2D(max_err_idx,j,Y_DIM_1);
			std::cout << (int)HARDWARE_OUTPUT(hardware_out_y, idx, max_err_idx, j);
			// [idx / PORT_VEC_SIZE][idx % PORT_VEC_SIZE];
			// std::cout << (int)hardware_out_y[IDX2D(max_err_idx,j,Y_DIM_1)];
			// std::cout << (int)hardware_out_y[max_err_idx][j];
			std::cout << "\t\t <--> \t\t" << (int)REFERENCE_OUTPUT(max_err_idx,j) << std::endl;
		}

		for (int i = 0; i < Y_DIM_0; i++) {
			for (int j = 0; j < Y_DIM_1; j++) {
				int idx = IDX2D(i,j,Y_DIM_1);
				// if (HARDWARE_OUTPUT(hardware_out_y, i, j)
				// [idx / PORT_VEC_SIZE][idx % PORT_VEC_SIZE] != 0) {
					std::cout << "(" << i << "," << j << ") => \t";
					// std::cout << hardware_out_y[IDX2D(i,j,Y_DIM_1)];
					// std::cout << hardware_out_y[i][j];
					std::cout << (int)HARDWARE_OUTPUT(hardware_out_y, idx, i, j);
					// [idx / PORT_VEC_SIZE][idx % PORT_VEC_SIZE];
					std::cout << "\t ?= \t" << REFERENCE_OUTPUT(i,j) << std::endl;
				// }
			}
		}
	}
	return 0;
}
