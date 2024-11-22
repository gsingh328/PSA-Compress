#include "hls_stream.h"
#include "resmlp.h"
#include "resmlp_model.h"


void read_data(
	hls::stream<model_act_t> &in,
	hls::stream<model_act_t> &strm_in) {

	// #pragma HLS DATAFLOW
	#pragma HLS INLINE off

in_buff_seq_lp:
	for (unsigned int i = 0; i < IN_SEQ_N; i++) {
	in_buff_vec_lp:
		for (unsigned int j = 0; j < IN_VEC_N; j++) {
		// Read data at half rate (this better matches the II to rest of the blocks)
		#pragma HLS PIPELINE II=2
			model_act_t x;
			in >> x;
			strm_in << x;
		}
	}
}


void write_data(
	hls::stream<model_act_t> &strm_out,
	hls::stream<model_act_t> &out) {

	// #pragma HLS DATAFLOW
	#pragma HLS INLINE off

out_buff_seq_lp:
	for (unsigned int i = 0; i < OUT_SEQ_N; i++) {
	out_buff_vec_lp:
		for (unsigned int j = 0; j < OUT_VEC_N; j++) {
			model_act_t y;
			strm_out >> y;
			out << y;
		}
	}
}


void krnl_resmlp(
	hls::stream<model_act_t> &in,
	hls::stream<model_act_t> &out
	) {

	// Free running kernel
	#pragma HLS interface ap_ctrl_none port=return

	#pragma HLS DATAFLOW

	hls::stream<model_act_t> strm_in;
	hls::stream<model_act_t> strm_out;
	// There is about a 500 cycle difference between the II of read_data and
	// the first layer in resmlp_model (embed), so have a buffer of 1024
	#pragma HLS STREAM variable=strm_in depth=BASE_FIFO_DEPTH
	#pragma HLS STREAM variable=strm_out depth=BASE_FIFO_DEPTH

	read_data(in, strm_in);
	resmlp_model(strm_in, strm_out);
	write_data(strm_out, out);
}
