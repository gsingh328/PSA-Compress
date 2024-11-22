#ifndef RESMLP_INPUT_LAYER_H
#define RESMLP_INPUT_LAYER_H

#include "accel_common.h"
#include "ld_buffer_tasks.h"
#include "resmlp_embed.h"


void start_task(
	const accel_io_vec_t *x_vec,
	accel_io_t io_buff[IO_BUFF_DIM_0][IO_BUFF_DIM_1]
	) {

	#pragma HLS INLINE off
	// #pragma HLS DATAFLOW

	copy_to_buff_par<accel_io_vec_t, accel_io_t, IO_BUFF_DIM_0, IO_BUFF_DIM_1>(
		x_vec,
		io_buff,
		0,
		RESMLP_IN_DIM_0,
		RESMLP_IN_DIM_1
	);
}


void resmlp_input_layer(
	const accel_io_vec_t *x_vec,
	const accel_w_vec_t *w_vec,
	const accel_b_vec_t *b_vec,
	const accel_s_vec_t *s_vec,
	accel_io_t io_buff[IO_BUFF_N][IO_BUFF_DIM_0][IO_BUFF_DIM_1],
    accel_io_t res_buff[IO_BUFF_DIM_0][IO_BUFF_DIM_1],
	accel_w_t w_buff[WBS_BUFF_N][W_BUFF_DIM_0][W_BUFF_DIM_1],
	accel_b_t b_buff[WBS_BUFF_N][BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	accel_s_t s_buff[WBS_BUFF_N][BS_BUFF_DIM_0][BS_BUFF_DIM_1],
    accel_w_t w_aff_buff[WBS_AFF_BUFF_N][W_AFF_BUFF_DIM_0][W_AFF_BUFF_DIM_1],
    accel_b_t b_aff_buff[WBS_AFF_BUFF_N][BS_AFF_BUFF_DIM_0][BS_AFF_BUFF_DIM_1],
    accel_s_t s_aff_buff[WBS_AFF_BUFF_N][BS_AFF_BUFF_DIM_0][BS_AFF_BUFF_DIM_1],
    const dim_t io_x_idx,
    const dim_t io_y_idx,
    const dim_t lin_wbs_idx,
    const dim_t aff_wbs_idx_1,
	const dim_t aff_wbs_idx_2,
	const dim_t aff_wbs_idx_3
	) {

    #pragma HLS INLINE

    start_task(
		x_vec,
		io_buff[io_x_idx]
	);

	ld_linear_buff_task(
		w_vec, 						// External Port to Weights
		b_vec, 						// External Port to Biases
		s_vec, 						// External Port to Scalars
		RM_EM_L0_W_OFFSET, 			// Weight Offset to read weights from DRAM
		RM_EM_L0_BS_OFFSET, 		// Weight Offset to read biases from DRAM
		RM_EM_L0_BS_OFFSET, 		// Weight Offset to read scalars from DRAM
		RM_EM_L0_W_DIM_0, 			// Weight Dimension 0
		RM_EM_L0_W_DIM_1, 			// Weight Dimension 1 (implied size for biases and scalar)
		w_buff[lin_wbs_idx], 		// Weight Buffer to store it in
		b_buff[lin_wbs_idx], 		// Bias buffer to store it in
		s_buff[lin_wbs_idx] 		// Scalar buffer to store it in
	);

	// =========================================================================

	processor(
		io_buff,					// IO Buffer Array
		res_buff,					// Residual IO Array
		io_x_idx, 					// Input X Index	(which IO buff to use)
		io_y_idx,					// Output Y Index	(which IO buff to use)
		w_buff[lin_wbs_idx], 	    // Weight Buffer
		b_buff[lin_wbs_idx], 	    // Bias Buffer
		s_buff[lin_wbs_idx], 		// Scalar Buffer
		RM_EM_L0_W_DIM_0,			// All Input Features
		RESMLP_IN_DIM_1, 			// Batch Dimension
		RM_EM_L0_W_DIM_1, 			// Output Features
		true,						// Add Bias
		false, 						// ReLU
		false, 						// Residual Add,
		false, 						// Transpose Input
		false, 						// Transpose Output
		true,						// Write to Residual

		false, 						        // Apply Pre-Affine
		w_aff_buff[aff_wbs_idx_1], 	    // Pre-Affine Weight
		b_aff_buff[aff_wbs_idx_1], 	    // Pre-Affine Bias
		s_aff_buff[aff_wbs_idx_1], 	    // Pre-Affine Scalar
		false, 						        // Apply Post-Affine
		w_aff_buff[aff_wbs_idx_2],     // Post-Affine Weight
		b_aff_buff[aff_wbs_idx_2],     // Post-Affine Bias
		s_aff_buff[aff_wbs_idx_2],     // Post-Affine Scalar
		false, 						        // Apply Res-Affine
		w_aff_buff[aff_wbs_idx_3],     // Res-Affine Weight
		b_aff_buff[aff_wbs_idx_3],     // Res-Affine Bias
		s_aff_buff[aff_wbs_idx_3] 	    // Res-Affine Scalar
	);
}


#endif // RESMLP_INPUT_LAYER_H
