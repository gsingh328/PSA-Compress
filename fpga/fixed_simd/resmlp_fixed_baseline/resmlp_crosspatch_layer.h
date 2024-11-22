#ifndef RESMLP_CROSSPATCH_LAYER_H
#define RESMLP_CROSSPATCH_LAYER_H


#include "resmlp_crosspatch.h"
#include "accel_common.h"
#include "ld_buffer_tasks.h"


void resmlp_crosspatch_layer(
	const accel_w_vec_t *w_vec,
	const accel_b_vec_t *b_vec,
	const accel_s_vec_t *s_vec,
	const accel_w_vec_t *w_aff_vec,
	const accel_b_vec_t *b_aff_vec,
	const accel_s_vec_t *s_aff_vec,
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
	const dim_t aff_wbs_idx_3,
	const len_t layer_fc_w_offset,
	const len_t layer_fc_bs_offset,
	const len_t layer_aff_w_offset,
	const len_t layer_aff_bs_offset
	) {

	#pragma HLS INLINE

	ld_processor_buff_task(
		w_vec, 						// External Port to Weights for Linear Params
		b_vec, 						// External Port to Biases for Linear Params
		s_vec, 						// External Port to Scalars for Linear Params
		w_aff_vec, 					// External Port to Weights for affine params
		b_aff_vec, 					// External Port to Biases for affine params
		s_aff_vec, 					// External Port to Scalars for affine params

		RM_CP_FC_W_OFFSET+layer_fc_w_offset,
		RM_CP_FC_BS_OFFSET+layer_fc_bs_offset,
		RM_CP_FC_BS_OFFSET+layer_fc_bs_offset,
		RM_CP_FC_W_DIM_0, 			// Weight Dimension 0
		RM_CP_FC_W_DIM_1, 			// Weight Dimension 1 (implied size for biases and scalar)
		w_buff[lin_wbs_idx], 		// Weight Buffer to store it in
		b_buff[lin_wbs_idx], 		// Bias buffer to store it in
		s_buff[lin_wbs_idx], 		// Scalar buffer to store it in

		RM_CP_AFF_1_W_OFFSET+layer_aff_w_offset,
		RM_CP_AFF_1_BS_OFFSET+layer_aff_bs_offset,
		RM_CP_AFF_1_BS_OFFSET+layer_aff_bs_offset,
		RM_CP_AFF_1_W_DIM_0, 				// Weight, Bias, Scalar Dimension 0
		w_aff_buff[aff_wbs_idx_1], 		// Weight Buffer to store it in
		b_aff_buff[aff_wbs_idx_1], 		// Bias buffer to store it in
		s_aff_buff[aff_wbs_idx_1], 		// Scalar buffer to store it in

		RM_CP_AFF_2_W_OFFSET+layer_aff_w_offset,
		RM_CP_AFF_2_BS_OFFSET+layer_aff_bs_offset,
		RM_CP_AFF_2_BS_OFFSET+layer_aff_bs_offset,
		RM_CP_AFF_2_W_DIM_0, 				// Weight, Bias, Scalar Dimension 0
		w_aff_buff[aff_wbs_idx_2], 	// Weight Buffer to store it in
		b_aff_buff[aff_wbs_idx_2], 	// Bias buffer to store it in
		s_aff_buff[aff_wbs_idx_2], 	// Scalar buffer to store it in

		RM_CP_RES_AFF_W_OFFSET+layer_aff_w_offset,
		RM_CP_RES_AFF_BS_OFFSET+layer_aff_bs_offset,
		RM_CP_RES_AFF_BS_OFFSET+layer_aff_bs_offset,
		RM_CP_RES_AFF_W_DIM_0, 				// Weight, Bias, Scalar Dimension 0
		w_aff_buff[aff_wbs_idx_3], 	// Weight Buffer to store it in
		b_aff_buff[aff_wbs_idx_3], 	// Bias buffer to store it in
		s_aff_buff[aff_wbs_idx_3] 		// Scalar buffer to store it in
	);

	// =========================================================================

	processor(
		io_buff,					// IO Buffer Array
		res_buff,					// Residual IO Array
		io_x_idx, 					// Input X Index	(which IO buff to use)
		io_y_idx,					// Output Y Index	(which IO buff to use)
		w_buff[lin_wbs_idx], 		// Weight Buffer
		b_buff[lin_wbs_idx], 		// Bias Buffer
		s_buff[lin_wbs_idx], 		// Scalar Buffer
		RM_CP_FC_W_DIM_0,			// All Input Features
		RM_CP_IN_DIM_0, 			// Batch Dimension
		RM_CP_FC_W_DIM_1, 			// Output Features
		true,						// Add Bias
		false, 						// ReLU
		true, 						// Residual Add,
		true, 						// Transpose Input
		true, 						// Transpose Output
		true,						// Write to Residual

		true, 								// Apply Pre-Affine
		w_aff_buff[aff_wbs_idx_1], 		// Pre-Affine Weight
		b_aff_buff[aff_wbs_idx_1], 		// Pre-Affine Bias
		s_aff_buff[aff_wbs_idx_1], 		// Pre-Affine Scalar
		true, 								// Apply Post-Affine
		w_aff_buff[aff_wbs_idx_2], 	// Post-Affine Weight
		b_aff_buff[aff_wbs_idx_2], 	// Post-Affine Bias
		s_aff_buff[aff_wbs_idx_2], 	// Post-Affine Scalar
		true, 								// Apply Res-Affine
		w_aff_buff[aff_wbs_idx_3], 	// Res-Affine Weight
		b_aff_buff[aff_wbs_idx_3], 	// Res-Affine Bias
		s_aff_buff[aff_wbs_idx_3] 		// Res-Affine Scalar
	);

}


#endif // RESMLP_CROSSPATCH_LAYER_H
