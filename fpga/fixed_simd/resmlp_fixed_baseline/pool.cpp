
#include "pool.h"
#include "utils.h"
#include <assert.h>
// #include <iostream>

// Will pool across dim 1
void avgpool(
	accel_io_t io_buff[IO_BUFF_N][IO_BUFF_DIM_0][IO_BUFF_DIM_1],
	const dim_t io_x_idx,
	const dim_t io_y_idx,
	const dim_t x_dim_0,
	const dim_t x_dim_1,
	const dim_t x_dim_1_lg_2
	) {

	#pragma HLS INLINE off

	// dependence comes from io_x_idx being equal to io_y_idx
	#pragma HLS dependence variable=io_buff type=inter false
	assert(io_x_idx!=io_y_idx);

	const dim_t x_dim_0_tiles = x_dim_0 / TILE_SIZE;
	const dim_t x_dim_1_tiles = x_dim_1 / TILE_SIZE;
	const dim_t n_vecs = (x_dim_0 * x_dim_1_tiles);

	// We are pooling across the x_dim_1 dimension that is tiled
	const dim_t n_acc_vecs = x_dim_1_tiles;

	static accel_io_t x_vec[TILE_SIZE];
	#pragma HLS array_partition variable=x_vec type=complete

	static accel_acc_t y_acc;
	static accel_acc_t out_y_acc;

	dim_t write_idx = 0;

	for (dim_t i_vec = 0; i_vec < n_vecs; i_vec++) {
		#pragma HLS PIPELINE II=1

		// In which order should we index (assuming no transpose)
		const dim_t x_tile_idx = i_vec;
		const bool should_reset_acc = (i_vec % n_acc_vecs) == 0;
		const bool completed_acc = ((i_vec + 1) % n_acc_vecs) == 0;

		// -----------------------------------------------------------------
		// Read a vector from buffer
		// -----------------------------------------------------------------
		for (dim_t i = 0; i < TILE_SIZE; i++) {
			#pragma HLS UNROLL

			x_vec[i] = io_buff[io_x_idx][x_tile_idx][i];
		}

		// -----------------------------------------------------------------
		// Accumulate
		// -----------------------------------------------------------------
		accel_acc_t vec_sum = 0;
		for (dim_t i = 0; i < TILE_SIZE; i++) {
			#pragma HLS UNROLL
			vec_sum += x_vec[i];
		}
		const accel_acc_t prev_y_acc = should_reset_acc ? (accel_acc_t) 0 : y_acc;
		y_acc = prev_y_acc + vec_sum;

		// Was this the last tile needed for accumulation
		// Copy to output copy of buffer
		if (completed_acc) {
			out_y_acc = y_acc;
		}

		ROUNDED_SHIFT(out_y_acc, x_dim_1_lg_2);
		// Skip-Saturate since average will never go above the range

		// --------------------------------
		// Write out the result
		// --------------------------------
		if (completed_acc) {
			io_buff[io_y_idx][write_idx][0] = (accel_io_t) out_y_acc;

			// Write 0 to rest of batch vector (i.e., zero padding batch dimension to TILE_SIZE)
			for (dim_t i = 1; i < TILE_SIZE; i++) {
				#pragma HLS UNROLL

				io_buff[io_y_idx][write_idx][i] = 0;
			}

			write_idx++;
		}
	}
}
