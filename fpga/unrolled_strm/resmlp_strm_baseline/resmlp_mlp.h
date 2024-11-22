#ifndef RESMLP_MLP_H
#define RESMLP_MLP_H

#include "resmlp.h"
#include "resmlp_utils.h"
#include "hls_stream.h"
#include "resmlp_extra_layers.h"


void resmlp_mlp(
	hls::stream<model_act_t> &strm_in,
	const model_w_t linear_1_w[HIDDEN_N][EMBED_N],
	const model_b_t linear_1_b[HIDDEN_N],
	const model_s_t linear_1_s[HIDDEN_N],
	const model_w_t linear_2_w[EMBED_N][HIDDEN_N],
	const model_b_t linear_2_b[EMBED_N],
	const model_s_t linear_2_s[EMBED_N],
	const model_w_t res_w[EMBED_N],
	const model_s_t res_s[EMBED_N],
	hls::stream<model_act_t> &strm_out
	) {

	#pragma HLS INLINE
	// #pragma HLS DATAFLOW

	hls::stream<model_act_t> strm_res;
	hls::stream<model_act_t> strm_matmul_in;
	// Due to MLP being all row major
	// an output "vector" go through the entire both matmuls
	// residual will have two tasks (each having two subtasks) infront of it
	// Be extra safe here since not having enough depth will cause a stall
	#pragma HLS STREAM variable=strm_res depth=((EMBED_N + HIDDEN_N) + (HIDDEN_N + EMBED_N) + RESIDUAL_DBG_PAD)
	#pragma HLS STREAM variable=strm_matmul_in depth=BASE_FIFO_DEPTH
	split_strm<SEQ_N, EMBED_N>(
		strm_in,
		strm_res,
		strm_matmul_in
	);

	hls::stream<model_act_t> strm_matmul_1_out;
	#pragma HLS STREAM variable=strm_matmul_1_out depth=BASE_FIFO_DEPTH
	matmul<SEQ_N, EMBED_N, HIDDEN_N, true>(
		strm_matmul_in,
		linear_1_w,
		linear_1_b,
		linear_1_s,
		strm_matmul_1_out
	);

	hls::stream<model_act_t> strm_matmul_2_out;
	#pragma HLS STREAM variable=strm_matmul_2_out depth=BASE_FIFO_DEPTH
	matmul<SEQ_N, HIDDEN_N, EMBED_N>(
		strm_matmul_1_out,
		linear_2_w,
		linear_2_b,
		linear_2_s,
		strm_matmul_2_out
	);

	rescale_add_residual<SEQ_N, EMBED_N>(
		strm_matmul_2_out,
		strm_res,
		res_w,
		res_s,
		strm_out
	);
}


#endif // RESMLP_MLP_H
