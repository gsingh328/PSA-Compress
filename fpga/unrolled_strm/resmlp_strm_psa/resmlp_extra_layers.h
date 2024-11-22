#ifndef RESMLP_EXTRA_LAYERS_H
#define RESMLP_EXTRA_LAYERS_H

#include "hls_stream.h"
#include "resmlp.h"
#include "resmlp_utils.h"


template<unsigned int x_col, unsigned int x_row, unsigned int y_row, bool relu=false>
void matmul(
	hls::stream<model_act_t> &strm_in,
	const model_w_t w[y_row][x_row],
	const model_b_t b[y_row],
	const model_s_t s[y_row],
	hls::stream<model_act_t> &strm_out
	) {

	#pragma HLS INLINE off

	// #pragma HLS bind_storage variable=w type=ROM_1P
	// #pragma HLS array_partition variable=w dim=1 type=cyclic factor=2
	#pragma HLS array_reshape variable=w dim=2 type=complete

x_col_lp:
	for (unsigned int x_c = 0; x_c < x_col; x_c++) {
	#pragma HLS DATAFLOW

		model_act_t x_vec[x_row];

	// Read in a vector
	x_vec_rd_lp:
		for (unsigned int x_r = 0; x_r < x_row; x_r++) {
		#pragma HLS pipeline II=1
			strm_in >> x_vec[x_r];
		}
	// Reuse that vector for MatMul
	y_row_lp:
		for (unsigned int y_r = 0; y_r < y_row; y_r++) {
		#pragma HLS pipeline II=1

			matmul_acc_t acc = 0;
			for (unsigned int x_r = 0; x_r < x_row; x_r++) {
			#pragma HLS unroll
				acc = acc + (x_vec[x_r] * w[y_r][x_r]);
			}

			ROUNDED_SHIFT(acc, s[y_r]);
			acc += b[y_r];
			if (relu) {
				CLIP(acc, (matmul_acc_t) 0, (matmul_acc_t) ACT_MAX);
			} else {
				CLIP(acc, (matmul_acc_t) ACT_MIN, (matmul_acc_t) ACT_MAX);
			}

			model_act_t y = acc;
			strm_out << y;
		}
	}
}


template<unsigned int x_col, unsigned int x_row, unsigned int y_row>
void matmul_partial_unroll(
	hls::stream<model_act_t> &strm_in,
	const model_w_t w[y_row][x_row],
	const model_b_t b[y_row],
	const model_s_t s[y_row],
	hls::stream<model_act_t> &strm_out
	) {

	#pragma HLS INLINE off

x_col_lp:
	for (unsigned int x_c = 0; x_c < x_col; x_c++) {
	#pragma HLS DATAFLOW

		model_act_t x_vec[x_row];

	// Read in a vector
	x_vec_rd_lp:
		for (unsigned int x_r = 0; x_r < x_row; x_r++) {
		#pragma HLS pipeline II=1
			strm_in >> x_vec[x_r];
		}

	// Reuse that vector for MatMul
	y_row_lp:
		for (unsigned int y_r = 0; y_r < y_row; y_r++) {
		#pragma HLS loop_flatten off

			matmul_acc_t acc = 0;
			for (unsigned int x_r = 0; x_r < x_row; x_r++) {
			#pragma HLS pipeline II=1
				acc = acc + (x_vec[x_r] * w[y_r][x_r]);
			}

			ROUNDED_SHIFT(acc, s[y_r]);
			acc += b[y_r];
			CLIP(acc, (matmul_acc_t) ACT_MIN, (matmul_acc_t) ACT_MAX);

			model_act_t y = acc;
			strm_out << y;
		}
	}
}


template<unsigned int col, unsigned int row, unsigned int col_lg2>
void avgpool_col_with_shift(
	hls::stream<model_act_t> &strm_in,
	hls::stream<model_act_t> &strm_out
	) {

	#pragma HLS INLINE off
	// #pragma HLS DATAFLOW

	pool_acc_t acc[row];
// x_init_row_lp:
// 	for (unsigned int i_row = 0; i_row < row; i_row++) {
// 	#pragma HLS PIPELINE II=1
// 		acc[i_row] = 0;
// 	}

x_comp_col_lp:
	for (unsigned int i_col = 0; i_col < col; i_col++) {
	x_comp_row_lp:
		for (unsigned int i_row = 0; i_row < row; i_row++) {
		#pragma HLS PIPELINE II=1

			model_act_t x;
			strm_in >> x;

			pool_acc_t prev_acc = (i_col == 0) ? (pool_acc_t) 0 : acc[i_row];
			acc[i_row] = prev_acc + x;

			if (i_col == (col - 1)) {
				pool_acc_t tmpy = acc[i_row];
				ROUNDED_SHIFT(tmpy, col_lg2);
				// Skip-clip since average will never go above the range
				model_act_t y = tmpy;
				strm_out << y;
			}
		}
	}

// x_wr_row_lp:
// 	for (unsigned int i_row = 0; i_row < row; i_row++) {
// 	#pragma HLS PIPELINE II=1
// 		ROUNDED_SHIFT(acc[i_row], col_lg2);
// 		// Skip-clip since average will never go above the range
// 		model_act_t y = acc[i_row];
// 		strm_out << y;
// 	}
}


template<unsigned int dim0, unsigned int dim1>
void split_strm(
	hls::stream<model_act_t> &strm_in,
	hls::stream<model_act_t> &strm_out1,
	hls::stream<model_act_t> &strm_out2
	) {

	#pragma HLS INLINE off

	for (int i=0; i < dim0; i++) {
		for (int j=0; j < dim1; j++) {
		#pragma HLS loop_flatten
		#pragma HLS PIPELINE II=1
			model_act_t x;
			strm_in >> x;
			strm_out1 << x;
			strm_out2 << x;
		}
	}
}



static void _affine(
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


template<unsigned int col, unsigned int row>
void affine_transpose(
	hls::stream<model_act_t> &strm_in,
	const model_w_t affine_w[row],
	const model_b_t affine_b[row],
	const model_s_t affine_s[row],
	hls::stream<model_act_t> &strm_out
	) {

	#pragma HLS INLINE off
	#pragma HLS DATAFLOW

	// Due to transpose will need a full copy of input
	model_act_t in_buff[col][row];

col_lp:
	for (unsigned int i_col = 0; i_col < col; i_col++) {
	row_lp:
		for (unsigned int i_row = 0; i_row < row; i_row++) {
			model_act_t x, aff_y;
			strm_in >> x;

			_affine(
				x,
				affine_w[i_row],
				affine_b[i_row],
				affine_s[i_row],
				aff_y
			);

			in_buff[i_col][i_row] = aff_y;
		}
	}

tnp_row_lp:
	for (unsigned int i_row = 0; i_row < row; i_row++) {
	tnp_col_lp:
		for (unsigned int i_col = 0; i_col < col; i_col++) {
			strm_out << in_buff[i_col][i_row];
		}
	}
}


template<unsigned int col, unsigned int row>
void transpose_affine(
	hls::stream<model_act_t> &strm_in,
	const model_w_t affine_w[row],
	const model_b_t affine_b[row],
	const model_s_t affine_s[row],
	hls::stream<model_act_t> &strm_out
	) {

	#pragma HLS INLINE off
	#pragma HLS DATAFLOW

	// Due to transpose will need a full copy of input
	model_act_t in_buff[col][row];

tnp_row_lp:
	for (unsigned int i_row = 0; i_row < row; i_row++) {
	tnp_col_lp:
		for (unsigned int i_col = 0; i_col < col; i_col++) {
			strm_in >> in_buff[i_col][i_row];
		}
	}

col_lp:
	for (unsigned int i_col = 0; i_col < col; i_col++) {
	row_lp:
		for (unsigned int i_row = 0; i_row < row; i_row++) {
			model_act_t x, aff_y;
			x = in_buff[i_col][i_row];

			_affine(
				x,
				affine_w[i_row],
				affine_b[i_row],
				affine_s[i_row],
				aff_y
			);

			strm_out << aff_y;
		}
	}
}


template<unsigned int dim0, unsigned int dim1>
void rescale_add_residual(
	hls::stream<model_act_t> &strm_x,
	hls::stream<model_act_t> &strm_res,
	const model_w_t res_w[dim1],
	const model_s_t res_s[dim1],
	hls::stream<model_act_t> &strm_out
	) {

	#pragma HLS INLINE off

	for (int i=0; i < dim0; i++) {
		for (int j=0; j < dim1; j++) {
		#pragma HLS loop_flatten
		#pragma HLS PIPELINE II=1
			model_act_t x, res_x;
			strm_x >> x;
			strm_res >> res_x;

			affine_acc_t acc = res_x * res_w[j];
			ROUNDED_SHIFT(acc, res_s[j]);
			CLIP(acc, (affine_acc_t) ACT_MIN, (affine_acc_t) ACT_MAX);

			add_clip_acc_t res_acc = x + (add_clip_acc_t) acc;
			CLIP(res_acc, (add_clip_acc_t) ACT_MIN, (add_clip_acc_t) ACT_MAX);

			model_act_t y = res_acc;
			strm_out << y;
		}
	}
}


#endif // RESMLP_EXTRA_LAYERS_H
