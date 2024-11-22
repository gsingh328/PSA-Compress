#ifndef LD_BUFFER_TASKS_H
#define LD_BUFFER_TASKS_H


#include "dma_utils.h"
#include "accel_common.h"


void ld_linear_buff_sub_task(
	const accel_b_vec_t *b_vec,
	const accel_s_vec_t *s_vec,

	// Linear Params
	const len_t b_offset,
	const len_t s_offset,
	const dim_t dim_1,
	accel_b_t b_buff[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	accel_s_t s_buff[BS_BUFF_DIM_0][BS_BUFF_DIM_1]
	) {

	#pragma HLS INLINE
	// Bias and Scalar will load in parallel
	// #pragma HLS DATAFLOW

	copy_to_buff_par<accel_b_vec_t, accel_b_t, BS_BUFF_DIM_0, BS_BUFF_DIM_1>(
		b_vec,
		b_buff,
		b_offset,
		1,
		dim_1
	);

	copy_to_buff_par<accel_s_vec_t, accel_s_t, BS_BUFF_DIM_0, BS_BUFF_DIM_1>(
		s_vec,
		s_buff,
		s_offset,
		1,
		dim_1
	);
}


void ld_linear_buff_task(
	const accel_w_vec_t *w_vec,
	const accel_b_vec_t *b_vec,
	const accel_s_vec_t *s_vec,
	const len_t w_offset,
	const len_t b_offset,
	const len_t s_offset,
	const dim_t dim_0,
	const dim_t dim_1,
	accel_w_t w_buff[W_BUFF_DIM_0][W_BUFF_DIM_1],
	accel_b_t b_buff[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	accel_s_t s_buff[BS_BUFF_DIM_0][BS_BUFF_DIM_1]
	) {

	#pragma HLS INLINE off
	// Weights, Bias+Scalar will load sequentially

	copy_to_buff_par<accel_w_vec_t, accel_w_t, W_BUFF_DIM_0, W_BUFF_DIM_1>(
		w_vec,
		w_buff,
		w_offset,
		dim_0,
		dim_1
	);

	ld_linear_buff_sub_task(
		b_vec,
		s_vec,

		// Linear Params
		b_offset,
		s_offset,
		dim_1,
		b_buff,
		s_buff
	);
}


void ld_affine_buff_task(
	const accel_w_vec_t *w_aff_vec,
	const accel_b_vec_t *b_aff_vec,
	const accel_s_vec_t *s_aff_vec,
	const len_t w_offset,
	const len_t b_offset,
	const len_t s_offset,
	const dim_t len,
	accel_w_t w_aff_buff[W_AFF_BUFF_DIM_0][W_AFF_BUFF_DIM_1],
	accel_b_t b_aff_buff[BS_AFF_BUFF_DIM_0][BS_AFF_BUFF_DIM_1],
	accel_s_t s_aff_buff[BS_AFF_BUFF_DIM_0][BS_AFF_BUFF_DIM_1]
	) {

	#pragma HLS INLINE off
	// All three parameters will load in parallel
	// #pragma HLS DATAFLOW

	copy_to_buff_par<accel_w_vec_t, accel_w_t, W_AFF_BUFF_DIM_0, W_AFF_BUFF_DIM_1>(
		w_aff_vec,
		w_aff_buff,
		w_offset,
		1,
		len
	);

	copy_to_buff_par<accel_b_vec_t, accel_b_t, BS_AFF_BUFF_DIM_0, BS_AFF_BUFF_DIM_1>(
		b_aff_vec,
		b_aff_buff,
		b_offset,
		1,
		len
	);

	copy_to_buff_par<accel_s_vec_t, accel_s_t, BS_AFF_BUFF_DIM_0, BS_AFF_BUFF_DIM_1>(
		s_aff_vec,
		s_aff_buff,
		s_offset,
		1,
		len
	);
}


void ld_processor_affine_sub_task(
	const accel_w_vec_t *w_aff_vec,
	const accel_b_vec_t *b_aff_vec,
	const accel_s_vec_t *s_aff_vec,

	// Affine Params 1
	const len_t w_aff_1_offset,
	const len_t b_aff_1_offset,
	const len_t s_aff_1_offset,
	const dim_t aff_1_dim_0,
	accel_w_t w_aff_1_buff[W_AFF_BUFF_DIM_0][W_AFF_BUFF_DIM_1],
	accel_b_t b_aff_1_buff[BS_AFF_BUFF_DIM_0][BS_AFF_BUFF_DIM_1],
	accel_s_t s_aff_1_buff[BS_AFF_BUFF_DIM_0][BS_AFF_BUFF_DIM_1],

	// Affine Params 2
	const len_t w_aff_2_offset,
	const len_t b_aff_2_offset,
	const len_t s_aff_2_offset,
	const dim_t aff_2_dim_0,
	accel_w_t w_aff_2_buff[W_AFF_BUFF_DIM_0][W_AFF_BUFF_DIM_1],
	accel_b_t b_aff_2_buff[BS_AFF_BUFF_DIM_0][BS_AFF_BUFF_DIM_1],
	accel_s_t s_aff_2_buff[BS_AFF_BUFF_DIM_0][BS_AFF_BUFF_DIM_1],

	// Affine Params 3
	const len_t w_aff_3_offset,
	const len_t b_aff_3_offset,
	const len_t s_aff_3_offset,
	const dim_t aff_3_dim_0,
	accel_w_t w_aff_3_buff[W_AFF_BUFF_DIM_0][W_AFF_BUFF_DIM_1],
	accel_b_t b_aff_3_buff[BS_AFF_BUFF_DIM_0][BS_AFF_BUFF_DIM_1],
	accel_s_t s_aff_3_buff[BS_AFF_BUFF_DIM_0][BS_AFF_BUFF_DIM_1]
	) {

	#pragma HLS INLINE off

	ld_affine_buff_task(
		w_aff_vec,
		b_aff_vec,
		s_aff_vec,
		w_aff_1_offset,
		b_aff_1_offset,
		s_aff_1_offset,
		aff_1_dim_0,
		w_aff_1_buff,
		b_aff_1_buff,
		s_aff_1_buff
	);

	ld_affine_buff_task(
		w_aff_vec,
		b_aff_vec,
		s_aff_vec,
		w_aff_2_offset,
		b_aff_2_offset,
		s_aff_2_offset,
		aff_2_dim_0,
		w_aff_2_buff,
		b_aff_2_buff,
		s_aff_2_buff
	);

	ld_affine_buff_task(
		w_aff_vec,
		b_aff_vec,
		s_aff_vec,
		w_aff_3_offset,
		b_aff_3_offset,
		s_aff_3_offset,
		aff_3_dim_0,
		w_aff_3_buff,
		b_aff_3_buff,
		s_aff_3_buff
	);
}


void ld_processor_linear_sub_task(
	const accel_w_vec_t *w_vec,
	const accel_b_vec_t *b_vec,
	const accel_s_vec_t *s_vec,
	const len_t w_offset,
	const len_t b_offset,
	const len_t s_offset,
	const dim_t dim_0,
	const dim_t dim_1,
	accel_w_t w_buff[W_BUFF_DIM_0][W_BUFF_DIM_1],
	accel_b_t b_buff[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	accel_s_t s_buff[BS_BUFF_DIM_0][BS_BUFF_DIM_1]
	) {

	#pragma HLS INLINE off
	// Weights, Bias+Scalar will load sequentially
	// #pragma HLS DATAFLOW

	copy_to_buff_par<accel_w_vec_t, accel_w_t, W_BUFF_DIM_0, W_BUFF_DIM_1>(
		w_vec,
		w_buff,
		w_offset,
		dim_0,
		dim_1
	);

	ld_linear_buff_sub_task(
		b_vec,
		s_vec,

		// Linear Params
		b_offset,
		s_offset,
		dim_1,
		b_buff,
		s_buff
	);
}


void ld_processor_buff_task(
	const accel_w_vec_t *w_vec,
	const accel_b_vec_t *b_vec,
	const accel_s_vec_t *s_vec,
	const accel_w_vec_t *w_aff_vec,
	const accel_b_vec_t *b_aff_vec,
	const accel_s_vec_t *s_aff_vec,

	// Linear Params
	const len_t w_offset,
	const len_t b_offset,
	const len_t s_offset,
	const dim_t dim_0,
	const dim_t dim_1,
	accel_w_t w_buff[W_BUFF_DIM_0][W_BUFF_DIM_1],
	accel_b_t b_buff[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	accel_s_t s_buff[BS_BUFF_DIM_0][BS_BUFF_DIM_1],

	// Affine Params 1
	const len_t w_aff_1_offset,
	const len_t b_aff_1_offset,
	const len_t s_aff_1_offset,
	const dim_t aff_1_dim_0,
	accel_w_t w_aff_1_buff[W_AFF_BUFF_DIM_0][W_AFF_BUFF_DIM_1],
	accel_b_t b_aff_1_buff[BS_AFF_BUFF_DIM_0][BS_AFF_BUFF_DIM_1],
	accel_s_t s_aff_1_buff[BS_AFF_BUFF_DIM_0][BS_AFF_BUFF_DIM_1],

	// Affine Params 2
	const len_t w_aff_2_offset,
	const len_t b_aff_2_offset,
	const len_t s_aff_2_offset,
	const dim_t aff_2_dim_0,
	accel_w_t w_aff_2_buff[W_AFF_BUFF_DIM_0][W_AFF_BUFF_DIM_1],
	accel_b_t b_aff_2_buff[BS_AFF_BUFF_DIM_0][BS_AFF_BUFF_DIM_1],
	accel_s_t s_aff_2_buff[BS_AFF_BUFF_DIM_0][BS_AFF_BUFF_DIM_1],

	// Affine Params 3
	const len_t w_aff_3_offset,
	const len_t b_aff_3_offset,
	const len_t s_aff_3_offset,
	const dim_t aff_3_dim_0,
	accel_w_t w_aff_3_buff[W_AFF_BUFF_DIM_0][W_AFF_BUFF_DIM_1],
	accel_b_t b_aff_3_buff[BS_AFF_BUFF_DIM_0][BS_AFF_BUFF_DIM_1],
	accel_s_t s_aff_3_buff[BS_AFF_BUFF_DIM_0][BS_AFF_BUFF_DIM_1]
	) {

	#pragma HLS INLINE
	// #pragma HLS DATAFLOW

	ld_linear_buff_task(
		w_vec,
		b_vec,
		s_vec,
		w_offset,
		b_offset,
		s_offset,
		dim_0,
		dim_1,
		w_buff,
		b_buff,
		s_buff
	);

	ld_processor_affine_sub_task(
		w_aff_vec,
		b_aff_vec,
		s_aff_vec,

		w_aff_1_offset,
		b_aff_1_offset,
		s_aff_1_offset,
		aff_1_dim_0,
		w_aff_1_buff,
		b_aff_1_buff,
		s_aff_1_buff,

		w_aff_2_offset,
		b_aff_2_offset,
		s_aff_2_offset,
		aff_2_dim_0,
		w_aff_2_buff,
		b_aff_2_buff,
		s_aff_2_buff,

		w_aff_3_offset,
		b_aff_3_offset,
		s_aff_3_offset,
		aff_3_dim_0,
		w_aff_3_buff,
		b_aff_3_buff,
		s_aff_3_buff
	);
}


#endif // LD_BUFFER_TASKS_H
