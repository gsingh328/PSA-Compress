#ifndef RESMLP_LAYERS_H
#define RESMLP_LAYERS_H

#include "resmlp.h"
#include "resmlp_block_macros.h"

#if(HIDDEN_N==192)
#include "quant_params.h"
#endif

#if(HIDDEN_N==96)
#include "quant_params_h_mul_1.h"
#endif


void resmlp_blocks(
	hls::stream<model_act_t> &strm_embed,
	hls::stream<model_act_t> &strm_blocks
	) {

	#pragma HLS INLINE
	// #pragma HLS DATAFLOW

	hls::stream<model_act_t> strm_tkn_0;
	hls::stream<model_act_t> strm_mlp_0;
	#pragma HLS STREAM variable=strm_tkn_0 depth=BASE_FIFO_DEPTH
	#pragma HLS STREAM variable=strm_mlp_0 depth=BASE_FIFO_DEPTH
	CALL_RESMLP_TOKEN_MIXER(0, strm_embed, strm_tkn_0);
	CALL_RESMLP_CROSS_CHANNEL(0, strm_tkn_0, strm_mlp_0);

	hls::stream<model_act_t> strm_tkn_1;
	hls::stream<model_act_t> strm_mlp_1;
	#pragma HLS STREAM variable=strm_tkn_1 depth=BASE_FIFO_DEPTH
	#pragma HLS STREAM variable=strm_mlp_1 depth=BASE_FIFO_DEPTH
	CALL_RESMLP_TOKEN_MIXER(1, strm_mlp_0, strm_tkn_1);
	CALL_RESMLP_CROSS_CHANNEL(1, strm_tkn_1, strm_mlp_1);

	hls::stream<model_act_t> strm_tkn_2;
	hls::stream<model_act_t> strm_mlp_2;
	#pragma HLS STREAM variable=strm_tkn_2 depth=BASE_FIFO_DEPTH
	#pragma HLS STREAM variable=strm_mlp_2 depth=BASE_FIFO_DEPTH
	CALL_RESMLP_TOKEN_MIXER(2, strm_mlp_1, strm_tkn_2);
	CALL_RESMLP_CROSS_CHANNEL(2, strm_tkn_2, strm_mlp_2);

	hls::stream<model_act_t> strm_tkn_3;
	#pragma HLS STREAM variable=strm_tkn_3 depth=BASE_FIFO_DEPTH
	CALL_RESMLP_TOKEN_MIXER(3, strm_mlp_2, strm_tkn_3);
	CALL_RESMLP_CROSS_CHANNEL(3, strm_tkn_3, strm_blocks);
}


#endif //RESMLP_LAYERS_H
