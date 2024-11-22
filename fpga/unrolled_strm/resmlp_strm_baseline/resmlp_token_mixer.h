#ifndef RESMLP_TOKEN_MIXER_H
#define RESMLP_TOKEN_MIXER_H


#include "resmlp.h"
#include "resmlp_utils.h"
#include "hls_stream.h"
#include "resmlp_extra_layers.h"


static void scale(
	const model_act_t x,
	const model_w_t w,
	const model_s_t s,
	model_act_t &y
	) {
	#pragma HLS INLINE
	// #pragma HLS FUNCTION_INSTANTIATE variable=w,s
	affine_acc_t acc = x * w;
	ROUNDED_SHIFT(acc, s);
	CLIP(acc, (affine_acc_t) ACT_MIN, (affine_acc_t) ACT_MAX);
	y = acc;
}


static void affine(
	const model_act_t x,
	const model_w_t w,
	const model_b_t b,
	const model_s_t s,
	model_act_t &y
	) {
	#pragma HLS INLINE
	// #pragma HLS FUNCTION_INSTANTIATE variable=w,b,s
	affine_acc_t acc = x * w;
	ROUNDED_SHIFT(acc, s);
	acc = acc + b;
	CLIP(acc, (affine_acc_t) ACT_MIN, (affine_acc_t) ACT_MAX);
	y = acc;
}


void resmlp_token_mixer(
	hls::stream<model_act_t> &strm_in,
	const model_w_t affine_1_w[EMBED_N],
	const model_b_t affine_1_b[EMBED_N],
	const model_s_t affine_1_s[EMBED_N],
	const model_w_t linear_w[SEQ_N][SEQ_N],
	const model_b_t linear_b[SEQ_N],
	const model_s_t linear_s[SEQ_N],
	const model_w_t affine_2_w[EMBED_N],
	const model_b_t affine_2_b[EMBED_N],
	const model_s_t affine_2_s[EMBED_N],
	const model_w_t res_w[EMBED_N],
	const model_s_t res_s[EMBED_N],
	hls::stream<model_act_t> &strm_out
	) {

	#pragma HLS INLINE
	// #pragma HLS DATAFLOW

	hls::stream<model_act_t> strm_res;
	hls::stream<model_act_t> strm_aff_1_in;
	// the residual path will require large depth due to the transpose operations
	// basically full copy of inputs * 2
	// additionally there will be a matmul with a SEQ_N length
	// Add extra to be extra safe here since not having enough depth will cause a stall
	#pragma HLS STREAM variable=strm_res depth=(2*(SEQ_N+(EMBED_N*SEQ_N)) + RESIDUAL_DBG_PAD)
	#pragma HLS STREAM variable=strm_aff_1_in depth=BASE_FIFO_DEPTH
	split_strm<SEQ_N, EMBED_N>(
		strm_in,
		strm_res,
		strm_aff_1_in
	);

	hls::stream<model_act_t> strm_aff_1_out;
	#pragma HLS STREAM variable=strm_aff_1_out depth=BASE_FIFO_DEPTH
	affine_transpose<SEQ_N, EMBED_N>(
		strm_aff_1_in,
		affine_1_w,
		affine_1_b,
		affine_1_s,
		strm_aff_1_out
	);

	hls::stream<model_act_t> strm_matmul_out;
	// the next affine will consume at higher rate
	#pragma HLS STREAM variable=strm_matmul_out depth=BASE_FIFO_DEPTH
	matmul<EMBED_N, SEQ_N, SEQ_N>(
		strm_aff_1_out,
		linear_w,
		linear_b,
		linear_s,
		strm_matmul_out
	);

	hls::stream<model_act_t> strm_aff_2_out;
	// the next residual will consume at higher rate
	#pragma HLS STREAM variable=strm_aff_2_out depth=BASE_FIFO_DEPTH
	transpose_affine<SEQ_N, EMBED_N>(
		strm_matmul_out,
		affine_2_w,
		affine_2_b,
		affine_2_s,
		strm_aff_2_out
	);

	rescale_add_residual<SEQ_N, EMBED_N>(
		strm_aff_2_out,
		strm_res,
		res_w,
		res_s,
		strm_out
	);
}

#endif // RESMLP_TOKEN_MIXER_H
