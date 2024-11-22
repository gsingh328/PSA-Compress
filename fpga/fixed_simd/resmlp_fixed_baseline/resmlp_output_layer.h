#ifndef RESMLP_OUTPUT_LAYER_H
#define RESMLP_OUTPUT_LAYER_H


#include "accel_common.h"
#include "ld_buffer_tasks.h"
#include "resmlp_model.h"


void end_task(
	const accel_io_t io_buff[IO_BUFF_DIM_0][IO_BUFF_DIM_1],
	accel_io_vec_t *y_vec
	) {

	#pragma HLS INLINE off

	write_from_buff_par<accel_io_t, accel_io_vec_t, IO_BUFF_DIM_0, IO_BUFF_DIM_1>(
		io_buff,
		y_vec,
		0,
		RESMLP_OUT_DIM_0,
		RESMLP_OUT_DIM_1
	);
}


void resmlp_output_layer(
	accel_io_vec_t *y_vec,
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
	const dim_t lin_wbs_idx_2,		// not using this, added just for compatibility
	const dim_t aff_2_wbs_idx_1,	// not using this, added just for compatibility
	const dim_t aff_2_wbs_idx_2,	// not using this, added just for compatibility
	const dim_t aff_2_wbs_idx_3, 	// not using this, added just for compatibility
	const len_t layer_fc_w_offset,
	const len_t layer_fc_bs_offset
	) {


    #pragma HLS INLINE


    ld_processor_buff_task(
		w_vec, 						// External Port to Weights for Linear Params
		b_vec, 						// External Port to Biases for Linear Params
		s_vec, 						// External Port to Scalars for Linear Params
		w_aff_vec, 					// External Port to Weights for affine params
		b_aff_vec, 					// External Port to Biases for affine params
		s_aff_vec, 					// External Port to Scalars for affine params

		RM_O_LAYER_FC_W_OFFSET+layer_fc_w_offset,
		RM_O_LAYER_FC_BS_OFFSET+layer_fc_bs_offset,
		RM_O_LAYER_FC_BS_OFFSET+layer_fc_bs_offset,
		RM_OUTP_FC_W_DIM_0, 				// Weight Dimension 0
		RM_OUTP_FC_W_DIM_1, 				// Weight Dimension 1 (implied size for biases and scalar)
		w_buff[lin_wbs_idx_1], 				// Weight Buffer to store it in
		b_buff[lin_wbs_idx_1], 				// Bias buffer to store it in
		s_buff[lin_wbs_idx_1], 				// Scalar buffer to store it in

		0, 									// Weight Offset to read weights from DRAM
		0, 									// Weight Offset to read biases from DRAM
		0, 									// Weight Offset to read scalars from DRAM
		0, 									// Weight, Bias, Scalar Dimension 0
		w_aff_buff[aff_1_wbs_idx_1], 		// Weight Buffer to store it in
		b_aff_buff[aff_1_wbs_idx_1], 		// Bias buffer to store it in
		s_aff_buff[aff_1_wbs_idx_1], 		// Scalar buffer to store it in

		0, 									// Weight Offset to read weights from DRAM
		0, 									// Weight Offset to read biases from DRAM
		0, 									// Weight Offset to read scalars from DRAM
		0, 									// Weight, Bias, Scalar Dimension 0
		w_aff_buff[aff_1_wbs_idx_2], 	// Weight Buffer to store it in
		b_aff_buff[aff_1_wbs_idx_2], 	// Bias buffer to store it in
		s_aff_buff[aff_1_wbs_idx_2], 	// Scalar buffer to store it in

		0,
		0,
		0,
		0, 									// Weight, Bias, Scalar Dimension 0
		w_aff_buff[aff_1_wbs_idx_3], 	// Weight Buffer to store it in
		b_aff_buff[aff_1_wbs_idx_3], 	// Bias buffer to store it in
		s_aff_buff[aff_1_wbs_idx_3] 		// Scalar buffer to store it in
	);

	avgpool(
		io_buff,					// IO Buffer Array
		io_x_idx_1, 				// Input X Index	(which IO buff to use)
		io_y_idx_1,					// Output Y Index	(which IO buff to use)
		RM_OUTP_IN_DIM_0,			// Input Features
		RESMLP_IN_DIM_1,			// Batch Dimension (this is the one that is pooled)
		RM_OUTP_IN_DIM_1_LG2		// LOG2 of the above value
	);

	// =========================================================================

	processor(
		io_buff,					// IO Buffer Array
		res_buff,					// Residual IO Array
		io_x_idx_2, 				// Input X Index	(which IO buff to use)
		io_y_idx_2,					// Output Y Index	(which IO buff to use)
		w_buff[lin_wbs_idx_1], 		// Weight Buffer
		b_buff[lin_wbs_idx_1], 		// Bias Buffer
		s_buff[lin_wbs_idx_1], 		// Scalar Buffer
		RM_OUTP_FC_W_DIM_0,			// All Input Features
		1, 							// Batch Dimension
		CEIL_MULTIPLE(RM_OUTP_FC_W_DIM_1, TILE_SIZE),
		true,						// Add Bias
		false, 						// ReLU
		false, 						// Residual Add,
		false, 						// Transpose Input
		false, 						// Transpose Output
		false,						// Write to Residual

		false, 								// Apply Pre-Affine
		w_aff_buff[aff_1_wbs_idx_1], 	// Pre-Affine Weight
		b_aff_buff[aff_1_wbs_idx_1], 	// Pre-Affine Bias
		s_aff_buff[aff_1_wbs_idx_1], 	// Pre-Affine Scalar
		false, 								// Apply Post-Affine
		w_aff_buff[aff_1_wbs_idx_2], 	// Post-Affine Weight
		b_aff_buff[aff_1_wbs_idx_2], 	// Post-Affine Bias
		s_aff_buff[aff_1_wbs_idx_2], 	// Post-Affine Scalar
		false, 								// Apply Res-Affine
		w_aff_buff[aff_1_wbs_idx_3], 	// Res-Affine Weight
		b_aff_buff[aff_1_wbs_idx_3], 	// Res-Affine Bias
		s_aff_buff[aff_1_wbs_idx_3] 	// Res-Affine Scalar
	);

	// =========================================================================

	end_task(
		io_buff[io_y_idx_2],
		y_vec
	);
}


#endif // RESMLP_OUTPUT_LAYER_H
