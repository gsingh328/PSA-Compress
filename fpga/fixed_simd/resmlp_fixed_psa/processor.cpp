#include "processor.h"
#include "utils.h"
#include <assert.h>
// #include <iostream>
//#include <stdlib.h>
#include "utils.h"


// #define USE_DSP_PACKING


void dsp_int8_packed_mul(
	ap_int<8> x,
	ap_int<8> w1,
	ap_int<8> w2,
	ap_int<16> &y1,
	ap_int<16> &y2
	) {

	#pragma HLS INLINE

	// Port A
	ap_int<27> dsp_A = 0;
	dsp_A.range(25,18) =  w1.range(7,0);
	dsp_A[26] = w1[7];

	// Port B
	ap_int<27> dsp_B = w2;

	// Port D
	ap_int<18> dsp_D = x;

	// Compute
	ap_int<34> dsp_O = (dsp_A + dsp_B) * dsp_D;
	#pragma HLS bind_op variable=dsp_O op=mul impl=dsp

	// Add the rounding term during extraction
	y1.range(15,0) = dsp_O.range(33,18) + dsp_O[17];
	y2.range(15,0) = dsp_O.range(15,0);
}


void affine(
	const accel_io_t x_vec[TILE_SIZE],
	const dim_t ifeat,
	const bool apply_affine,
	const accel_w_t w_aff[W_AFF_BUFF_DIM_0][W_AFF_BUFF_DIM_1],
	const accel_b_t b_aff[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	const accel_s_t s_aff[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	accel_io_t y_vec[TILE_SIZE]
	) {

	#pragma HLS INLINE

	const dim_t aff_w_d0 = (ifeat / W_AFF_BUFF_DIM_1) % W_AFF_BUFF_DIM_0;
	const dim_t aff_w_d1 = ifeat % W_AFF_BUFF_DIM_1;
	const dim_t aff_bs_d0 = (ifeat / BS_AFF_BUFF_DIM_1) % BS_AFF_BUFF_DIM_0;
	const dim_t aff_bs_d1 = ifeat % BS_AFF_BUFF_DIM_1;

	const accel_w_t w_for_vec = w_aff[aff_w_d0][aff_w_d1];
	const accel_b_t b_for_vec = b_aff[aff_bs_d0][aff_bs_d1];
	const accel_s_t s_for_vec = s_aff[aff_bs_d0][aff_bs_d1];

	for (dim_t i = 0; i < TILE_SIZE; i++) {
	#pragma HLS UNROLL

		accel_io_t tmp;
		if (apply_affine) {
			accel_mul_t acc = x_vec[i] * w_for_vec;
			#pragma HLS BIND_OP variable=acc op=mul impl=dsp
			ROUNDED_SHIFT(acc, s_for_vec);
			acc += b_for_vec;
			CLIP(acc, ACCEL_IO_CLIP_MIN, ACCEL_IO_CLIP_MAX);
			tmp = (accel_io_t) acc;
		} else {
			tmp = x_vec[i];
		}

		y_vec[i] = tmp;
	}
}


void lerp_psa_lookup(
	const accel_io_t x_vec[TILE_SIZE],
	const bool apply_psa,
	const model_psa_t psa_lut_buff[PSA_LUT_BUFF_DIM_0][PSA_LUT_BUFF_DIM_1],
	const dim_t idx_offset,
	accel_io_t y_vec[TILE_SIZE]
	) {

	#pragma HLS INLINE

	for (dim_t i = 0; i < TILE_SIZE; i++) {
	#pragma HLS UNROLL

		model_psa_off_t tmpx = x_vec[i];
		model_psa_idx_t x_idx = (tmpx + (model_psa_off_t) PSA_IDX_OFFSET);

		// Split index into MSB and LSB portion for LERP
		psa_msb_idx_t x_msb = x_idx.range(LERP_PSA_BITS-1, LERP_PSA_LSB_BITS);
		psa_lsb_idx_t x_lsb = x_idx.range(LERP_PSA_LSB_BITS-1, 0);

		// Clip the x_msb to be within the range (can skip due to subspliting)
		// CLIP_MAX(x_msb, ((psa_msb_idx_t)(LERP_PSA_N - 2)));

		model_psa_t y0 = psa_lut_buff[idx_offset + x_msb][i];
		model_psa_t y1 = psa_lut_buff[idx_offset + x_msb + 1][i];

		psa_lerp_t y = (y1 - y0) * x_lsb;
		#pragma HLS BIND_OP variable=y op=mul impl=fabric

		// y >>= LERP_PSA_LSB_BITS;
		ROUNDED_SHIFT(y, LERP_PSA_LSB_BITS);

		y += y0;

		accel_io_t out_y = (accel_io_t) y;

		if (apply_psa) {
			y_vec[i] = out_y;
		} else {
			y_vec[i] = x_vec[i];
		}
	}
}


void res_add_and_affine(
	const accel_io_t y_vec[TILE_SIZE],
	const accel_io_t res_vec[TILE_SIZE],
	const dim_t ifeat,
	const bool add_residual,
	const bool apply_post_affine,
	const accel_w_t w_post_aff[W_AFF_BUFF_DIM_0][W_AFF_BUFF_DIM_1],
	const accel_b_t b_post_aff[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	const accel_s_t s_post_aff[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	const bool apply_res_affine,
	const accel_w_t w_res_aff[W_AFF_BUFF_DIM_0][W_AFF_BUFF_DIM_1],
	const accel_b_t b_res_aff[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	const accel_s_t s_res_aff[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	accel_io_t y_aff_res_vec[TILE_SIZE]
	) {

	#pragma HLS INLINE

	const dim_t aff_w_d0 = (ifeat / W_AFF_BUFF_DIM_1) % W_AFF_BUFF_DIM_0;
	const dim_t aff_w_d1 = ifeat % W_AFF_BUFF_DIM_1;
	const dim_t aff_bs_d0 = (ifeat / BS_AFF_BUFF_DIM_1) % BS_AFF_BUFF_DIM_0;
	const dim_t aff_bs_d1 = ifeat % BS_AFF_BUFF_DIM_1;

	const accel_w_t res_aff_w_for_vec = w_res_aff[aff_w_d0][aff_w_d1];
	const accel_b_t res_aff_b_for_vec = b_res_aff[aff_bs_d0][aff_bs_d1];
	const accel_s_t res_aff_s_for_vec = s_res_aff[aff_bs_d0][aff_bs_d1];

	const accel_w_t o_aff_w_for_vec = w_post_aff[aff_w_d0][aff_w_d1];
	const accel_b_t o_aff_b_for_vec = b_post_aff[aff_bs_d0][aff_bs_d1];
	const accel_s_t o_aff_s_for_vec = s_post_aff[aff_bs_d0][aff_bs_d1];

	for (dim_t j = 0; j < TILE_SIZE; j++) {
		#pragma HLS UNROLL

		accel_io_t aff_y;
		if (apply_post_affine) {
			accel_mul_t acc = y_vec[j] * o_aff_w_for_vec;
			#pragma HLS BIND_OP variable=acc op=mul impl=dsp
			ROUNDED_SHIFT(acc, o_aff_s_for_vec);
			acc += o_aff_b_for_vec;
			CLIP(acc, ACCEL_IO_CLIP_MIN, ACCEL_IO_CLIP_MAX);
			aff_y = (accel_io_t) acc;
		} else {
			aff_y = y_vec[j];
		}

		accel_io_t res_y;
		if (apply_res_affine) {
			accel_mul_t acc = res_vec[j] * res_aff_w_for_vec;
			#pragma HLS BIND_OP variable=acc op=mul impl=dsp
			ROUNDED_SHIFT(acc, res_aff_s_for_vec);
			acc += res_aff_b_for_vec;
			CLIP(acc, ACCEL_IO_CLIP_MIN, ACCEL_IO_CLIP_MAX);
			res_y = (accel_io_t) acc;
		} else {
			res_y = res_vec[j];
		}

		accel_res_add_t tmp_y = aff_y;

		// Residual
		tmp_y += (add_residual) ? res_y : ((accel_io_t) 0);

		CLIP(tmp_y, ACCEL_IO_CLIP_MIN, ACCEL_IO_CLIP_MAX);
		accel_io_t out_y = (accel_io_t) tmp_y;

		y_aff_res_vec[j] = out_y;
	}
}


void processor(
	// Linear Params
	accel_io_t io_buff[IO_BUFF_N][IO_BUFF_DIM_0][IO_BUFF_DIM_1],
	accel_io_t res_buff[IO_BUFF_DIM_0][IO_BUFF_DIM_1],
	const dim_t io_x_idx,
	const dim_t io_y_idx,
	const accel_w_t w[W_BUFF_DIM_0][W_BUFF_DIM_1],
	const accel_b_t b[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	const accel_s_t s[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	const dim_t x_dim_0,
	const dim_t x_dim_1,
	const dim_t w_dim_1,
	const bool add_bias,
	const bool apply_relu,
	const bool add_residual,
	const bool transpose_input,
	const bool transpose_output,
	const bool write_res,

	// Affine Params
	const bool apply_pre_affine,
	const accel_w_t w_pre_aff[W_AFF_BUFF_DIM_0][W_AFF_BUFF_DIM_1],
	const accel_b_t b_pre_aff[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	const accel_s_t s_pre_aff[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	const bool apply_post_affine,
	const accel_w_t w_post_aff[W_AFF_BUFF_DIM_0][W_AFF_BUFF_DIM_1],
	const accel_b_t b_post_aff[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	const accel_s_t s_post_aff[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	const bool apply_res_affine,
	const accel_w_t w_res_aff[W_AFF_BUFF_DIM_0][W_AFF_BUFF_DIM_1],
	const accel_b_t b_res_aff[BS_BUFF_DIM_0][BS_BUFF_DIM_1],
	const accel_s_t s_res_aff[BS_BUFF_DIM_0][BS_BUFF_DIM_1],

	// PSA Params
	const bool apply_psa,
	const model_psa_t psa_lut_buff[PSA_LUT_BUFF_DIM_0][PSA_LUT_BUFF_DIM_1]
	) {

	#pragma HLS INLINE off
	// #pragma HLS DATAFLOW

	// dependence comes from io_x_idx being equal to io_y_idx
	#pragma HLS dependence variable=io_buff type=inter false
	assert(io_x_idx!=io_y_idx);

	// dependence comes from when we are writing to res_buff but also adding res_buff to our output
	#pragma HLS dependence variable=res_buff type=inter false
	// assert((add_residual ^ write_res) || ((!add_residual) && (!write_res)));

	const dim_t x_dim_0_tiles = DIV_ROUNDUP(x_dim_0, TILE_SIZE);
	const dim_t x_dim_1_tiles = DIV_ROUNDUP(x_dim_1, TILE_SIZE);
	const dim_t w_dim_1_tiles = DIV_ROUNDUP(w_dim_1, TILE_SIZE);

	// How many total vectors of TILE_SIZE do we have to iterate through including all batches
	// Add TILE_SIZE cycles to read/transpose in x
	// Add TILE_SIZE cycles to do PSA (operates on a transposed vector)
	// Add TILE_SIZE cycles to quantized out the accumulation tile
	// Add TILE_SIZE cycles to write/transpose out y
	const dim_t n_vecs = (x_dim_0 * x_dim_1_tiles * w_dim_1_tiles) + (4 * TILE_SIZE);
	// const dim_t n_vecs = x_dim_0 * x_dim_1_tiles * w_dim_1_tiles;

	// After how many vectors should we reset our accumulation buffer
	const dim_t n_acc_vecs = x_dim_0;

	// Active buffer of accumulating y_tile
	static accel_acc_t y_tile[TILE_SIZE][TILE_SIZE];
	#pragma HLS array_partition variable=y_tile type=complete

	// Output copy of y_tile
	static accel_acc_t out_y_tile[TILE_SIZE][TILE_SIZE];
	#pragma HLS array_partition variable=out_y_tile type=complete

	static accel_io_t out_y_vec[TILE_SIZE];
	#pragma HLS array_partition variable=out_y_vec type=complete

	static accel_io_t x_vec[TILE_SIZE];
	static accel_w_t w_vec[TILE_SIZE];
	static accel_io_t res_vec[TILE_SIZE];
	static accel_io_t y_vec[TILE_SIZE];
	#pragma HLS array_partition variable=x_vec type=complete
	#pragma HLS array_partition variable=w_vec type=complete
	#pragma HLS array_partition variable=res_vec type=complete
	#pragma HLS array_partition variable=y_vec type=complete

	// Ping Pong buffer for reading x (needing to handle transposing)
	static accel_io_t x_aff_psa_tile[2][TILE_SIZE][TILE_SIZE];
	#pragma HLS array_partition variable=x_aff_psa_tile type=complete

	// Ping Pong buffer for doing PSA (it operated on a transposed input)
	static accel_io_t x_psa_mm_tile[2][TILE_SIZE][TILE_SIZE];
	#pragma HLS array_partition variable=x_psa_mm_tile type=complete

	// Ping Pong buffer for writing y (needing to handle transposing)
	static accel_io_t write_y_tile[2][TILE_SIZE][TILE_SIZE];
	#pragma HLS array_partition variable=write_y_tile type=complete

	// Buffers for storing affine outputs
	static accel_io_t x_aff_vec[TILE_SIZE];
	static accel_io_t y_aff_res_vec[TILE_SIZE];
	#pragma HLS array_partition variable=x_aff_vec type=complete
	#pragma HLS array_partition variable=y_aff_res_vec type=complete

	// Buffers for PSA input and output
	static accel_io_t x_psa_in_vec[TILE_SIZE];
	static accel_io_t x_psa_out_vec[TILE_SIZE];
	#pragma HLS array_partition variable=x_psa_in_vec type=complete
	#pragma HLS array_partition variable=x_psa_out_vec type=complete

	// dim_t out_vecs_remaining = 0;
	dim_t out_i_vec = 0;
	dim_t write_i_vec = 0;

vec_lp:
	for (dim_t i_vec = 0; i_vec < n_vecs; i_vec++) {
	#pragma HLS PIPELINE II=1
		// Helpful partial results used to calculate indices
		const dim_t i_vec_div_x_dim_0 = i_vec / x_dim_0;
		const dim_t i_vec_mod_x_dim_0 = i_vec % x_dim_0;
		const dim_t i_vec_div_TILE_SIZE = i_vec / TILE_SIZE;
		const dim_t i_vec_mod_TILE_SIZE = i_vec % TILE_SIZE;

		// Indices for Non-Transposed Input
		// When not transposing we index across the inputs, ie the x_d0 dimension
		const dim_t x_d0 = i_vec_mod_x_dim_0;
		const dim_t x_d1_vec = i_vec_div_x_dim_0 % x_dim_1_tiles;
		const dim_t nt_x_vec_idx = (x_d0 * x_dim_1_tiles) + x_d1_vec;

		// Indices for Transposed Input
		// Index so it handles reading in TILE_SIZE x TILE_SIZE and transposing it
		const dim_t x_d1 = (x_d1_vec * TILE_SIZE) + i_vec_mod_TILE_SIZE;
		const dim_t x_d0_vec = i_vec_div_TILE_SIZE % x_dim_0_tiles;
		const dim_t t_x_vec_idx = (x_d1 * x_dim_0_tiles) + x_d0_vec;

		const dim_t x_vec_idx = transpose_input ? t_x_vec_idx : nt_x_vec_idx;

		// Ping pong indexing into buffers b/w affine and PSA
		const dim_t x_aff_psa_wr_idx = i_vec_div_TILE_SIZE % 2;
		const dim_t x_aff_psa_rd_idx = (x_aff_psa_wr_idx + 1) % 2;

		// Ping pong indexing into buffers b/w PSA and MatMul
		const dim_t x_psa_mm_wr_idx = ((i_vec - TILE_SIZE) / TILE_SIZE) % 2;
		const dim_t x_psa_mm_rd_idx = (x_psa_mm_wr_idx + 1) % 2;

		// Our matrix multiplication flags will be 2*TILE_SIZE behind
		const dim_t i_vec_subt2 = i_vec - (2 * TILE_SIZE);
		const dim_t i_vec_subt2_mod_x_dim_0 = i_vec_subt2 % x_dim_0;
		const dim_t i_vec_subt2_div_x_dim_0 = i_vec_subt2 / x_dim_0;
		const bool reset_acc = (i_vec_subt2_mod_x_dim_0 == 0);
		const bool completed_acc = ((i_vec_subt2 + 1) % x_dim_0) == 0;

		// We want to start our output every x_dim_0 iterations for TILE_SIZE amount
		// make sure not to do it until the first accumulation has finished
		// since the matmul is TILE_SIZE behind, our output will also be TILE_SIZE behind
		const bool should_output = (i_vec_subt2_mod_x_dim_0 < TILE_SIZE) && (i_vec_subt2 >= x_dim_0);

		// Similar to above, but the writing of the output will be three TILE_SIZE's behind
		const dim_t i_vec_subt3 = i_vec - (3 * TILE_SIZE);
		const dim_t i_vec_subt3_mod_x_dim_0 = i_vec_subt3 % x_dim_0;
		const bool should_write = (i_vec_subt3_mod_x_dim_0 < TILE_SIZE) && (i_vec_subt3 >= x_dim_0);

		// -----------------------------------------------------------------
		// Read a vector from buffer
		// -----------------------------------------------------------------
		for (dim_t i = 0; i < TILE_SIZE; i++) {
		#pragma HLS UNROLL

			x_vec[i] = io_buff[io_x_idx][x_vec_idx][i];
		}

		// -----------------------------------------------------------------
		// 	Apply input affine
		// -----------------------------------------------------------------

		// If transposing our actual input feature can be for x_d1 or x_d0
		// Since affine is applied before transposing and matmul
		const dim_t input_ifeat = transpose_input ? x_d1 : x_d0;

		affine(x_vec, input_ifeat, apply_pre_affine, w_pre_aff, b_pre_aff, s_pre_aff, x_aff_vec);

		// -----------------------------------------------------------------
		// Read from vector into TILE_SIZE x TILE_SIZE buffer for transposing
		// -----------------------------------------------------------------
		for (dim_t i = 0; i < TILE_SIZE; i++) {
			#pragma HLS UNROLL

			x_aff_psa_tile[x_aff_psa_wr_idx][i_vec % TILE_SIZE][i] = x_aff_vec[i];

			// if (transpose_input) {
			// 	x_aff_psa_tile[x_aff_psa_wr_idx][i][i_vec % TILE_SIZE] = x_aff_vec[i];
			// } else {
			// 	x_aff_psa_tile[x_aff_psa_wr_idx][i_vec % TILE_SIZE][i] = x_aff_vec[i];
			// }
		}

		// -----------------------------------------------------------------
		// 	PSA
		// -----------------------------------------------------------------
		{
			// PSA will be working TILE_SIZE behind
			const dim_t i_vec_subt = i_vec - TILE_SIZE;
			const dim_t i_vec_subt_mod_x_dim_0 = i_vec_subt % x_dim_0;

			const dim_t psa_x_d0 = i_vec_subt_mod_x_dim_0;
			const dim_t psa_x_d1_vec = (i_vec_subt / x_dim_0) % x_dim_1_tiles;
			const dim_t psa_nt_x_vec_idx = (psa_x_d0 * x_dim_1_tiles) + psa_x_d1_vec;

			// Indices for Transposed Input
			// Index so it handles reading in TILE_SIZE x TILE_SIZE and transposing it
			const dim_t psa_x_d1 = (psa_x_d1_vec * TILE_SIZE) + (i_vec_subt % TILE_SIZE);

			// If transposing our actual input feature can be for x_d1 or x_d0
			// Since affine is applied before transposing and matmul
			const dim_t psa_input_ifeat = transpose_input ? psa_x_d1 : psa_x_d0;

			// Read a transposed vector for applying PSA
			for (dim_t i = 0; i < TILE_SIZE; i++) {
			#pragma HLS UNROLL

				// x_psa_in_vec[i] = x_aff_psa_tile[x_aff_psa_rd_idx][i][i_vec % TILE_SIZE];

				// we read in a transposed vector by default
				if (!transpose_input) {
					x_psa_in_vec[i] = x_aff_psa_tile[x_aff_psa_rd_idx][i][i_vec % TILE_SIZE];
				} else {
					x_psa_in_vec[i] = x_aff_psa_tile[x_aff_psa_rd_idx][i_vec % TILE_SIZE][i];
				}
			}

			// -----------------------------------------------------------------
			// LERP PSA
			// -----------------------------------------------------------------
			lerp_psa_lookup(
				x_psa_in_vec,
				apply_psa,
				psa_lut_buff,
				(psa_input_ifeat / TILE_SIZE) * LERP_PSA_N,
				x_psa_out_vec);

			// Write back our transposed output
			for (dim_t i = 0; i < TILE_SIZE; i++) {
			#pragma HLS UNROLL
				x_psa_mm_tile[x_psa_mm_wr_idx][i][i_vec % TILE_SIZE] = x_psa_out_vec[i];
			}
		}

		// -----------------------------------------------------------------
		// Read weights
		// -----------------------------------------------------------------

		// they are a 2*TILE_SIZE behind the inputs
		const dim_t x_d0_subt = i_vec_subt2_mod_x_dim_0;
		const dim_t w_d1_subt_vec = (i_vec_subt2_div_x_dim_0 / x_dim_1_tiles) % w_dim_1_tiles;
		const dim_t w_vec_idx = (x_d0_subt * w_dim_1_tiles) + w_d1_subt_vec;

		for (dim_t i = 0; i < TILE_SIZE; i++) {
		#pragma HLS UNROLL

			w_vec[i] = w[w_vec_idx][i];
		}

		// -----------------------------------------------------------------
		// Compute Multiplication (with dsp packing)
		// -----------------------------------------------------------------
		#ifndef USE_DSP_PACKING

		for (dim_t i = 0; i < TILE_SIZE; i++) {
		#pragma HLS UNROLL
			for (dim_t j = 0; j < TILE_SIZE; j++) {
			#pragma HLS UNROLL

				accel_acc_t current_y = x_psa_mm_tile[x_psa_mm_rd_idx][i_vec % TILE_SIZE][j] *
					w_vec[i];
				// accel_acc_t current_y = x_vec[j] * w_vec[i];

				accel_acc_t prev_y;
				if (reset_acc) {
					prev_y = 0;
				} else {
					prev_y = y_tile[i][j];
				}

				y_tile[i][j] = prev_y + current_y;

				// Was this the last tile needed for accumulation
				// Copy to output copy of buffer
				if (completed_acc) {
					out_y_tile[i][j] = y_tile[i][j];
				}
			}
		}

		#else

		for (dim_t i = 0; i < TILE_SIZE; i+=2) {
		#pragma HLS UNROLL
			for (dim_t j = 0; j < TILE_SIZE; j++) {
			#pragma HLS UNROLL

				const accel_io_t x = x_psa_mm_tile[x_psa_mm_rd_idx][i_vec % TILE_SIZE][j];
				const accel_io_t w1 = w_vec[i];
				const accel_io_t w2 = w_vec[i+1];

				ap_int<16> y1, y2;
				dsp_int8_packed_mul(x, w1, w2, y1, y2);

				accel_acc_t prev_y1, prev_y2;
				if (reset_acc) {
					prev_y1 = 0;
					prev_y2 = 0;
				} else {
					prev_y1 = y_tile[i][j];
					prev_y2 = y_tile[i+1][j];
				}

				y_tile[i][j] = prev_y1 + y1;
				y_tile[i+1][j] = prev_y2 + y2;

				// Was this the last tile needed for accumulation
				// Copy to output copy of buffer
				if (completed_acc) {
					out_y_tile[i][j] = y_tile[i][j];
					out_y_tile[i+1][j] = y_tile[i+1][j];
				}
			}
		}

		#endif

		// -----------------------------------------------------------------
		// Compute Bias Add + Requantization
		// -----------------------------------------------------------------
		{
			const dim_t out_i_vec_div_TILE_SIZE = out_i_vec / TILE_SIZE;

			// Row within TILE_SIZE x TILE_SIZE
			const dim_t y_wr_idx = out_i_vec_div_TILE_SIZE % 2;
			const dim_t write_vec_idx = out_i_vec % TILE_SIZE;

			// Get which tile are we writing to
			const dim_t write_x_d1_vec = out_i_vec_div_TILE_SIZE % x_dim_1_tiles;
			const dim_t write_w_d1_vec = out_i_vec_div_TILE_SIZE / x_dim_1_tiles;

			// Convert tile indices to element-wise index offset
			const dim_t write_w_d1 = (write_w_d1_vec * TILE_SIZE) + write_vec_idx;
			const dim_t param_w_d1 = write_w_d1 % w_dim_1;

			const dim_t bs_d0 = (param_w_d1 / BS_AFF_BUFF_DIM_1) % BS_AFF_BUFF_DIM_0;
			const dim_t bs_d1 = param_w_d1 % BS_AFF_BUFF_DIM_1;

			const accel_b_t b_for_vec = b[bs_d0][bs_d1];
			const accel_s_t s_for_vec = s[bs_d0][bs_d1] > 0 ? s[bs_d0][bs_d1] : (accel_s_t) 1;

			for (dim_t j = 0; j < TILE_SIZE; j++) {
				#pragma HLS UNROLL

				accel_acc_t acc = out_y_tile[write_vec_idx][j];

				ROUNDED_SHIFT(acc, s_for_vec);
				acc += (add_bias) ? b_for_vec : (accel_b_t) 0;

				CLIP(acc, ACCEL_IO_CLIP_MIN, ACCEL_IO_CLIP_MAX);
				accel_io_t out_y = (accel_io_t) acc;

				// Activation Function
				out_y = (apply_relu && (out_y < 0)) ? ((accel_io_t) 0) : out_y;

				out_y_vec[j] = out_y;
			}

			if (should_output) {
				for (dim_t j = 0; j < TILE_SIZE; j++) {
				#pragma HLS UNROLL

					write_y_tile[y_wr_idx][write_vec_idx][j] = out_y_vec[j];
				}
				out_i_vec++;
			}
		}

		// -----------------------------------------------------------------
		// Handle: transposing, output and residual affine, residual add
		// Also write out the final result to io_buff
		// -----------------------------------------------------------------
		{
			const dim_t write_i_vec_div_TILE_SIZE = write_i_vec / TILE_SIZE;

			// Row within TILE_SIZE x TILE_SIZE
			const dim_t y_wr_idx = write_i_vec_div_TILE_SIZE % 2;
			const dim_t write_vec_idx = write_i_vec % TILE_SIZE;

			// Get which tile are we writing to
			const dim_t write_x_d1_vec = write_i_vec_div_TILE_SIZE % x_dim_1_tiles;
			const dim_t write_w_d1_vec = write_i_vec_div_TILE_SIZE / x_dim_1_tiles;

			// Convert tile indices to element-wise index offset
			const dim_t write_w_d1 = (write_w_d1_vec * TILE_SIZE) + write_vec_idx;

			const dim_t nt_y_tile_idx = IDX2D(write_w_d1, write_x_d1_vec, x_dim_1_tiles);

			const dim_t idx0 = write_i_vec % x_dim_1;
			const dim_t idx1 = (write_i_vec / x_dim_1) % x_dim_1_tiles;
			const dim_t t_y_tile_idx = (idx0 * x_dim_0_tiles)  + idx1;

			const dim_t y_tile_idx = transpose_output ? t_y_tile_idx : nt_y_tile_idx;

			const dim_t output_ifeat = transpose_output ? (y_tile_idx / w_dim_1_tiles) : (write_w_d1);

			// ---------------------------------------
			// Read transposed
			// ---------------------------------------
			for (dim_t j = 0; j < TILE_SIZE; j++) {
			#pragma HLS UNROLL

				accel_io_t tmp_y;
				if (!transpose_output) {
					tmp_y = write_y_tile[y_wr_idx][write_vec_idx][j];
				} else {
					tmp_y = write_y_tile[y_wr_idx][j][write_vec_idx];
				}

				y_vec[j] = tmp_y;
			}

			// ---------------------------------------
			// Read residual
			// ---------------------------------------
			for (dim_t j = 0; j < TILE_SIZE; j++) {
			#pragma HLS UNROLL

				res_vec[j] = res_buff[y_tile_idx][j];
			}

			// ---------------------------------------
			// Residual affine -> Output affine -> Add
			// ---------------------------------------
			res_add_and_affine(y_vec, res_vec, output_ifeat, add_residual,
				apply_post_affine, w_post_aff, b_post_aff, s_post_aff,
				apply_res_affine, w_res_aff, b_res_aff, s_res_aff,
				y_aff_res_vec);

			// ---------------------------------------
			// Write out the result
			// ---------------------------------------
			if (should_write) {
				for (dim_t j = 0; j < TILE_SIZE; j++) {
				#pragma HLS UNROLL

					io_buff[io_y_idx][y_tile_idx][j] = y_aff_res_vec[j];
					if (write_res) {
						res_buff[y_tile_idx][j] = y_aff_res_vec[j];
					}
				}
				write_i_vec++;
			}
		}
	}
}
