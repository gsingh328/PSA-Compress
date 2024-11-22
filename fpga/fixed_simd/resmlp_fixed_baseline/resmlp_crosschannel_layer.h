#ifndef RESMLP_CROSSCHANNEL_LAYER_H
#define RESMLP_CROSSCHANNEL_LAYER_H


#include "resmlp_crosschannel.h"
#include "accel_common.h"
#include "ld_buffer_tasks.h"


void resmlp_crosschannel_layer(
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
    const dim_t io_x_idx_1,
    const dim_t io_y_idx_1,
    const dim_t lin_wbs_idx_1,
    const dim_t aff_1_wbs_idx_1,
	const dim_t aff_1_wbs_idx_2,
	const dim_t aff_1_wbs_idx_3,
    const dim_t io_x_idx_2,
    const dim_t io_y_idx_2,
    const dim_t lin_wbs_idx_2,
    const dim_t aff_2_wbs_idx_1,
	const dim_t aff_2_wbs_idx_2,
	const dim_t aff_2_wbs_idx_3,
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

		RM_CC_FC1_W_OFFSET+layer_fc_w_offset,
		RM_CC_FC1_BS_OFFSET+layer_fc_bs_offset,
		RM_CC_FC1_BS_OFFSET+layer_fc_bs_offset,
		RM_CC_FC1_W_DIM_0, 			// Weight Dimension 0
		RM_CC_FC1_W_DIM_1, 			// Weight Dimension 1 (implied size for biases and scalar)
		w_buff[lin_wbs_idx_1], 		// Weight Buffer to store it in
		b_buff[lin_wbs_idx_1], 		// Bias buffer to store it in
		s_buff[lin_wbs_idx_1], 		// Scalar buffer to store it in

		0, 							        // Weight Offset to read weights from DRAM
		0, 							        // Weight Offset to read biases from DRAM
		0, 							        // Weight Offset to read scalars from DRAM
		0, 							        // Weight, Bias, Scalar Dimension 0
		w_aff_buff[aff_1_wbs_idx_1], 	// Weight Buffer to store it in
		b_aff_buff[aff_1_wbs_idx_1], 	// Bias buffer to store it in
		s_aff_buff[aff_1_wbs_idx_1], 	// Scalar buffer to store it in

		0, 							        // Weight Offset to read weights from DRAM
		0, 							        // Weight Offset to read biases from DRAM
		0, 							        // Weight Offset to read scalars from DRAM
		0, 							        // Weight, Bias, Scalar Dimension 0
		w_aff_buff[aff_1_wbs_idx_2], 	// Weight Buffer to store it in
		b_aff_buff[aff_1_wbs_idx_2], 	// Bias buffer to store it in
		s_aff_buff[aff_1_wbs_idx_2], 	// Scalar buffer to store it in

		0, 							        // Weight Offset to read weights from DRAM
		0, 							        // Weight Offset to read biases from DRAM
		0, 							        // Weight Offset to read scalars from DRAM
		0, 							        // Weight, Bias, Scalar Dimension 0
		w_aff_buff[aff_1_wbs_idx_3], 	// Weight Buffer to store it in
		b_aff_buff[aff_1_wbs_idx_3], 	// Bias buffer to store it in
		s_aff_buff[aff_1_wbs_idx_3] 	// Scalar buffer to store it in
	);

	// =========================================================================

	processor(
		io_buff,					// IO Buffer Array
		res_buff,					// Residual IO Array
		io_x_idx_1, 							// Input X Index	(which IO buff to use)
		io_y_idx_1,							// Output Y Index	(which IO buff to use)
		w_buff[lin_wbs_idx_1], 					// Weight Buffer
		b_buff[lin_wbs_idx_1], 					// Bias Buffer
		s_buff[lin_wbs_idx_1], 					// Scalar Buffer
		RM_CC_FC1_W_DIM_0,			// All Input Features
		RESMLP_IN_DIM_1, 			// Batch Dimension
		RM_CC_FC1_W_DIM_1, 			// Output Features
		true,						// Add Bias
		true, 						// ReLU
		false, 						// Residual Add,
		false, 						// Transpose Input
		false, 						// Transpose Output
		false,						// Write to Residual

		false, 						        // Apply Pre-Affine
		w_aff_buff[aff_1_wbs_idx_1], 	// Pre-Affine Weight
		b_aff_buff[aff_1_wbs_idx_1], 	// Pre-Affine Bias
		s_aff_buff[aff_1_wbs_idx_1], 	// Pre-Affine Scalar
		false, 						        // Apply Post-Affine
		w_aff_buff[aff_1_wbs_idx_2], 	// Post-Affine Weight
		b_aff_buff[aff_1_wbs_idx_2], 	// Post-Affine Bias
		s_aff_buff[aff_1_wbs_idx_2], 	// Post-Affine Scalar
		false, 						        // Apply Res-Affine
		w_aff_buff[aff_1_wbs_idx_3], 	// Res-Affine Weight
		b_aff_buff[aff_1_wbs_idx_3], 	// Res-Affine Bias
		s_aff_buff[aff_1_wbs_idx_3] 	// Res-Affine Scalar
	);

	ld_processor_buff_task(
		w_vec, 						// External Port to Weights for Linear Params
		b_vec, 						// External Port to Biases for Linear Params
		s_vec, 						// External Port to Scalars for Linear Params
		w_aff_vec, 					// External Port to Weights for affine params
		b_aff_vec, 					// External Port to Biases for affine params
		s_aff_vec, 					// External Port to Scalars for affine params

		RM_CC_FC2_W_OFFSET+layer_fc_w_offset,
		RM_CC_FC2_BS_OFFSET+layer_fc_bs_offset,
		RM_CC_FC2_BS_OFFSET+layer_fc_bs_offset,
		RM_CC_FC2_W_DIM_0, 			// Weight Dimension 0
		RM_CC_FC2_W_DIM_1, 			// Weight Dimension 1 (implied size for biases and scalar)
		w_buff[lin_wbs_idx_2], 		// Weight Buffer to store it in
		b_buff[lin_wbs_idx_2], 		// Bias buffer to store it in
		s_buff[lin_wbs_idx_2], 		// Scalar buffer to store it in

		0, 							        // Weight Offset to read weights from DRAM
		0, 							        // Weight Offset to read biases from DRAM
		0, 							        // Weight Offset to read scalars from DRAM
		0, 							        // Weight, Bias, Scalar Dimension 0
		w_aff_buff[aff_2_wbs_idx_1], 	// Weight Buffer to store it in
		b_aff_buff[aff_2_wbs_idx_1], 	// Bias buffer to store it in
		s_aff_buff[aff_2_wbs_idx_1], 	// Scalar buffer to store it in

		0, 							        // Weight Offset to read weights from DRAM
		0, 							        // Weight Offset to read biases from DRAM
		0, 							        // Weight Offset to read scalars from DRAM
		0, 							        // Weight, Bias, Scalar Dimension 0
		w_aff_buff[aff_2_wbs_idx_2], 	// Weight Buffer to store it in
		b_aff_buff[aff_2_wbs_idx_2], 	// Bias buffer to store it in
		s_aff_buff[aff_2_wbs_idx_2], 	// Scalar buffer to store it in

		RM_CC_RES_AFF_W_OFFSET+layer_aff_w_offset,
		RM_CC_RES_AFF_BS_OFFSET+layer_aff_bs_offset,
		RM_CC_RES_AFF_BS_OFFSET+layer_aff_bs_offset,
		RM_CC_RES_AFF_W_DIM_0, 		        // Weight, Bias, Scalar Dimension 0
		w_aff_buff[aff_2_wbs_idx_3], 	// Weight Buffer to store it in
		b_aff_buff[aff_2_wbs_idx_3], 	// Bias buffer to store it in
		s_aff_buff[aff_2_wbs_idx_3] 	// Scalar buffer to store it in
	);

	// =========================================================================

	processor(
		io_buff,					// IO Buffer Array
		res_buff,					// Residual IO Array
		io_x_idx_2, 							// Input X Index	(which IO buff to use)
		io_y_idx_2,							// Output Y Index	(which IO buff to use)
		w_buff[lin_wbs_idx_2], 					// Weight Buffer
		b_buff[lin_wbs_idx_2], 					// Bias Buffer
		s_buff[lin_wbs_idx_2], 					// Scalar Buffer
		RM_CC_FC2_W_DIM_0,			// All Input Features
		RESMLP_IN_DIM_1, 			// Batch Dimension
		RM_CC_FC2_W_DIM_1, 			// Output Features
		true,						// Add Bias
		false, 						// ReLU
		true, 						// Residual Add,
		false, 						// Transpose Input
		false, 						// Transpose Output
		true,						// Write to Residual

		false, 						        // Apply Pre-Affine
		w_aff_buff[aff_2_wbs_idx_1], 	// Pre-Affine Weight
		b_aff_buff[aff_2_wbs_idx_1], 	// Pre-Affine Bias
		s_aff_buff[aff_2_wbs_idx_1], 	// Pre-Affine Scalar
		false, 						        // Apply Post-Affine
		w_aff_buff[aff_2_wbs_idx_2], 	// Post-Affine Weight
		b_aff_buff[aff_2_wbs_idx_2], 	// Post-Affine Bias
		s_aff_buff[aff_2_wbs_idx_2], 	// Post-Affine Scalar
		true, 						        // Apply Res-Affine
		w_aff_buff[aff_2_wbs_idx_3], 	// Res-Affine Weight
		b_aff_buff[aff_2_wbs_idx_3], 	// Res-Affine Bias
		s_aff_buff[aff_2_wbs_idx_3] 	// Res-Affine Scalar
	);

}


#endif // RESMLP_CROSSCHANNEL_LAYER_H
