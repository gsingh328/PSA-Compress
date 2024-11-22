#ifndef RESMLP_MODEL_H
#define RESMLP_MODEL_H

#include "hls_stream.h"
#include "resmlp_types.h"
#include "resmlp_blocks.h"
#include "resmlp_extra_layers.h"

#include "quant_params.h"
// #include "quant_params_h_mul_1.h"


// Wrapper functions for dataflow
void input_embed(
	hls::stream<model_act_t> &strm_in,
	hls::stream<model_act_t> &strm_embed
	) {

	#pragma HLS INLINE

	matmul<IN_SEQ_N, IN_VEC_N, EMBED_N>(
		strm_in,
		embed_linear_w,
		embed_linear_b,
		embed_linear_s,
		strm_embed
	);
}


// Wrapper functions for dataflow
void block_pool(
	hls::stream<model_act_t> &strm_block,
	hls::stream<model_act_t> &strm_pool
	) {

	#pragma HLS INLINE

	avgpool_col_with_shift<IN_SEQ_N, EMBED_N, SEQ_N_LG2>(
		strm_block,
		strm_pool
	);
}


// Wrapper functions for dataflow
void output_matmul(
	hls::stream<model_act_t> &strm_blocks,
	hls::stream<model_act_t> &strm_out
	) {

	#pragma HLS INLINE

	matmul_partial_unroll<OUT_SEQ_N, EMBED_N, OUT_VEC_N>(
		strm_blocks,
		classifier_linear_w,
		classifier_linear_b,
		classifier_linear_s,
		strm_out
	);
}


void resmlp_model(
	hls::stream<model_act_t> &strm_in,
	hls::stream<model_act_t> &strm_out
	) {

	#pragma HLS INLINE

	// -------------------------------------------------------------------------
	// Embedding
	// -------------------------------------------------------------------------
	hls::stream<model_act_t> strm_embed;
	#pragma HLS STREAM variable=strm_embed depth=BASE_FIFO_DEPTH
	input_embed(strm_in, strm_embed);

	// -------------------------------------------------------------------------
	// Blocks
	// -------------------------------------------------------------------------
	hls::stream<model_act_t> strm_blocks;
	#pragma HLS STREAM variable=strm_embed depth=BASE_FIFO_DEPTH
	resmlp_blocks(strm_embed, strm_blocks);

	// -------------------------------------------------------------------------
	// Pool
	// -------------------------------------------------------------------------

	hls::stream<model_act_t> strm_pool;
	#pragma HLS STREAM variable=strm_pool depth=BASE_FIFO_DEPTH
	block_pool(strm_blocks, strm_pool);

	// -------------------------------------------------------------------------
	// Output
	// -------------------------------------------------------------------------
	output_matmul(strm_pool, strm_out);
}


#endif //RESMLP_MODEL_H
