#include <stdint.h>
#include <iostream>

#include "resmlp.h"
#include "resmlp_model.h"
// #include "matmul_pe.h"
#include "processor.h"
#include "pool.h"
#include "utils.h"

#include "resmlp_input_layer.h"
#include "resmlp_crosspatch_layer.h"
#include "resmlp_crosschannel_layer.h"
#include "resmlp_output_layer.h"
#include "ld_buffer_tasks.h"

// -----------------------------------------------------------------------------
// Helper macros to define inputs into a layer
// -----------------------------------------------------------------------------
#define RESMLP_AXI_IMAGE_INPUT \
	x_vec
#define RESMLP_AXI_LABELS_OUTPUT \
	y_vec
#define RESMLP_AXI_LIN_WBS_INPUTS \
	w_vec, b_vec, s_vec
#define RESMLP_AXI_AFF_WBS_INPUTS \
	w_aff_vec, b_aff_vec, s_aff_vec
#define RESMLP_BUFF_INPUTS\
	io_buff, res_buff, w_buff, b_buff, s_buff, w_aff_buff, b_aff_buff, s_aff_buff
// #define RESMLP_BUFF_INDICES(IO_X_IDX, IO_Y_IDX, LIN_WBS_IDX, AFF_WBS_IDX_1, AFF_WBS_IDX_2, AFF_WBS_IDX_3)\
// 	IO_X_IDX, IO_Y_IDX, LIN_WBS_IDX, AFF_WBS_BASE_IDX, AFF_WBS_IDX_1, AFF_WBS_IDX_2, AFF_WBS_IDX_3
#define RESMLP_LAYER_PARAM_OFFSETS(N)\
	RM_LAYER_##N##_FC_W_OFFSET, RM_LAYER_##N##_FC_BS_OFFSET, RM_LAYER_##N##_AFF_W_OFFSET, RM_LAYER_##N##_AFF_BS_OFFSET

// Alternate b/w these two to ensure overlap
// b/w processor and ld_processor_buff_task

// io 0 -> 1, linear params in 0, affine params in (0, 1, 2)
#define RESMLP_LAYER_INDICES_CONFIG_1 0, 1, 0, 0, 1, 2
// io 1 -> 0, linear params in 1, affine params in (3, 4, 5)
#define RESMLP_LAYER_INDICES_CONFIG_2 1, 0, 1, 3, 4, 5
// -----------------------------------------------------------------------------


void krnl_resmlp(
	// Input
	const accel_io_vec_t *x_vec,
	// FC Layer Params
	const accel_w_vec_t *w_vec,
	const accel_b_vec_t *b_vec,
	const accel_s_vec_t *s_vec,
	// Affine Layer Params
	const accel_w_vec_t *w_aff_vec,
	const accel_b_vec_t *b_aff_vec,
	const accel_s_vec_t *s_aff_vec,
	// Output
	accel_io_vec_t *y_vec
	) {

	// #pragma HLS DATAFLOW

	#pragma HLS INTERFACE m_axi bundle=gmem0 port=x_vec max_widen_bitwidth=PORT_VEC_SIZE depth=((RESMLP_IN_DIM_0*RESMLP_IN_DIM_1)/PORT_VEC_SIZE)
	#pragma HLS INTERFACE m_axi bundle=gmem0 port=y_vec max_widen_bitwidth=PORT_VEC_SIZE depth=((RESMLP_OUT_PADDED_DIM_0*RESMLP_OUT_PADDED_DIM_1)/PORT_VEC_SIZE)

	#pragma HLS INTERFACE m_axi bundle=gmem1 port=w_vec max_widen_bitwidth=PORT_VEC_SIZE depth=(RM_FC_W_SIZE/PORT_VEC_SIZE) num_write_outstanding=1 max_write_burst_length=2
	#pragma HLS INTERFACE m_axi bundle=gmem1 port=b_vec max_widen_bitwidth=PORT_VEC_SIZE depth=(RM_FC_BS_SIZE/PORT_VEC_SIZE) num_write_outstanding=1 max_write_burst_length=2
	#pragma HLS INTERFACE m_axi bundle=gmem1 port=s_vec max_widen_bitwidth=PORT_VEC_SIZE depth=(RM_FC_BS_SIZE/PORT_VEC_SIZE) num_write_outstanding=1 max_write_burst_length=2

	#pragma HLS INTERFACE m_axi bundle=gmem0 port=w_aff_vec max_widen_bitwidth=PORT_VEC_SIZE depth=(RM_AFF_W_SIZE/PORT_VEC_SIZE) num_write_outstanding=1 max_write_burst_length=2
	#pragma HLS INTERFACE m_axi bundle=gmem0 port=b_aff_vec max_widen_bitwidth=PORT_VEC_SIZE depth=(RM_AFF_BS_SIZE/PORT_VEC_SIZE) num_write_outstanding=1 max_write_burst_length=2
	#pragma HLS INTERFACE m_axi bundle=gmem0 port=s_aff_vec max_widen_bitwidth=PORT_VEC_SIZE depth=(RM_AFF_BS_SIZE/PORT_VEC_SIZE) num_write_outstanding=1 max_write_burst_length=2

	static accel_io_t io_buff[IO_BUFF_N][IO_BUFF_DIM_0][IO_BUFF_DIM_1];
	#pragma HLS array_partition variable=io_buff dim=1 type=complete
	#pragma HLS array_partition variable=io_buff dim=3 type=complete

	static accel_io_t res_buff[IO_BUFF_DIM_0][IO_BUFF_DIM_1];
	#pragma HLS array_partition variable=res_buff dim=2 type=complete

	static accel_w_t w_buff[WBS_BUFF_N][W_BUFF_DIM_0][W_BUFF_DIM_1];
	static accel_b_t b_buff[WBS_BUFF_N][BS_BUFF_DIM_0][BS_BUFF_DIM_1];
	static accel_s_t s_buff[WBS_BUFF_N][BS_BUFF_DIM_0][BS_BUFF_DIM_1];
	#pragma HLS array_partition variable=w_buff dim=1 type=complete
	#pragma HLS array_partition variable=b_buff dim=1 type=complete
	#pragma HLS array_partition variable=s_buff dim=1 type=complete
	#pragma HLS array_reshape variable=w_buff dim=3 type=complete
	#pragma HLS array_reshape variable=b_buff dim=3 type=complete
	#pragma HLS array_reshape variable=s_buff dim=3 type=complete

	static accel_w_t w_aff_buff[WBS_AFF_BUFF_N][W_AFF_BUFF_DIM_0][W_AFF_BUFF_DIM_1];
	static accel_b_t b_aff_buff[WBS_AFF_BUFF_N][BS_AFF_BUFF_DIM_0][BS_AFF_BUFF_DIM_1];
	static accel_s_t s_aff_buff[WBS_AFF_BUFF_N][BS_AFF_BUFF_DIM_0][BS_AFF_BUFF_DIM_1];
	#pragma HLS array_partition variable=w_aff_buff dim=1 type=complete
	#pragma HLS array_partition variable=b_aff_buff dim=1 type=complete
	#pragma HLS array_partition variable=s_aff_buff dim=1 type=complete
	#pragma HLS array_reshape variable=w_aff_buff dim=3 type=complete
	#pragma HLS array_reshape variable=b_aff_buff dim=3 type=complete
	#pragma HLS array_reshape variable=s_aff_buff dim=3 type=complete


	#pragma HLS allocation function instances=processor limit=1
	// #pragma HLS allocation function instances=ld_linear_buff_task limit=1
	// #pragma HLS allocation function instances=ld_processor_affine_sub_task limit=1

	#pragma HLS bind_storage variable=io_buff type=RAM_S2P impl=bram
	#pragma HLS bind_storage variable=res_buff type=RAM_S2P impl=bram
	#pragma HLS bind_storage variable=w_buff type=RAM_1P impl=bram

	// =========================================================================
	// Input + Embed
	// =========================================================================

	resmlp_input_layer(
		RESMLP_AXI_IMAGE_INPUT,
		RESMLP_AXI_LIN_WBS_INPUTS,
		RESMLP_BUFF_INPUTS,
		RESMLP_LAYER_INDICES_CONFIG_1
	);

	// =========================================================================
	// Block 1
	// =========================================================================

	resmlp_crosspatch_layer(
		RESMLP_AXI_LIN_WBS_INPUTS,
		RESMLP_AXI_AFF_WBS_INPUTS,
		RESMLP_BUFF_INPUTS,
		RESMLP_LAYER_INDICES_CONFIG_2,
		RESMLP_LAYER_PARAM_OFFSETS(1)
	);

	resmlp_crosschannel_layer(
		RESMLP_AXI_LIN_WBS_INPUTS,
		RESMLP_AXI_AFF_WBS_INPUTS,
		RESMLP_BUFF_INPUTS,
		RESMLP_LAYER_INDICES_CONFIG_1,
		RESMLP_LAYER_INDICES_CONFIG_2,
		RESMLP_LAYER_PARAM_OFFSETS(1)
	);

	// =========================================================================
	// Block 2
	// =========================================================================

	resmlp_crosspatch_layer(
		RESMLP_AXI_LIN_WBS_INPUTS,
		RESMLP_AXI_AFF_WBS_INPUTS,
		RESMLP_BUFF_INPUTS,
		RESMLP_LAYER_INDICES_CONFIG_1,
		RESMLP_LAYER_PARAM_OFFSETS(2)
	);

	resmlp_crosschannel_layer(
		RESMLP_AXI_LIN_WBS_INPUTS,
		RESMLP_AXI_AFF_WBS_INPUTS,
		RESMLP_BUFF_INPUTS,
		RESMLP_LAYER_INDICES_CONFIG_2,
		RESMLP_LAYER_INDICES_CONFIG_1,
		RESMLP_LAYER_PARAM_OFFSETS(2)
	);

	// =========================================================================
	// Block 3
	// =========================================================================

	resmlp_crosspatch_layer(
		RESMLP_AXI_LIN_WBS_INPUTS,
		RESMLP_AXI_AFF_WBS_INPUTS,
		RESMLP_BUFF_INPUTS,
		RESMLP_LAYER_INDICES_CONFIG_2,
		RESMLP_LAYER_PARAM_OFFSETS(3)
	);

	resmlp_crosschannel_layer(
		RESMLP_AXI_LIN_WBS_INPUTS,
		RESMLP_AXI_AFF_WBS_INPUTS,
		RESMLP_BUFF_INPUTS,
		RESMLP_LAYER_INDICES_CONFIG_1,
		RESMLP_LAYER_INDICES_CONFIG_2,
		RESMLP_LAYER_PARAM_OFFSETS(3)
	);

	// =========================================================================
	// Block 4
	// =========================================================================

	resmlp_crosspatch_layer(
		RESMLP_AXI_LIN_WBS_INPUTS,
		RESMLP_AXI_AFF_WBS_INPUTS,
		RESMLP_BUFF_INPUTS,
		RESMLP_LAYER_INDICES_CONFIG_1,
		RESMLP_LAYER_PARAM_OFFSETS(4)
	);

	resmlp_crosschannel_layer(
		RESMLP_AXI_LIN_WBS_INPUTS,
		RESMLP_AXI_AFF_WBS_INPUTS,
		RESMLP_BUFF_INPUTS,
		RESMLP_LAYER_INDICES_CONFIG_2,
		RESMLP_LAYER_INDICES_CONFIG_1,
		RESMLP_LAYER_PARAM_OFFSETS(4)
	);

	// =========================================================================
	// Pool + Output
	// =========================================================================

	resmlp_output_layer(
		RESMLP_AXI_LABELS_OUTPUT,
		RESMLP_AXI_LIN_WBS_INPUTS,
		RESMLP_AXI_AFF_WBS_INPUTS,
		RESMLP_BUFF_INPUTS,
		RESMLP_LAYER_INDICES_CONFIG_2,
		RESMLP_LAYER_INDICES_CONFIG_1,
		RM_OUTP_FC_W_OFFSET,
		RM_OUTP_FC_BS_OFFSET
	);

	// NOTE: For debugging only!
	// end_task(
	// 	io_buff[1],
	// 	y_vec
	// );

}
