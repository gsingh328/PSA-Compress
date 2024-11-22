#ifndef DMA_UTILS_H
#define DMA_UTILS_H


#include <stdint.h>
#include "accel_common.h"


template<typename T>
void copy_to_buff_seq(
	const T *source_buff,
	T *destination_buff,
	const len_t source_offset,
	const len_t len,
	const len_t idx_offset
	) {

	#pragma HLS INLINE

	copy_to_buff_seq_lp:
	for (len_t i = 0; i < len; i++) {
		#pragma HLS loop_tripcount min=BUF_CP_SEQ_MIN_CNT max=BUF_CP_SEQ_MAX_CNT
		#pragma HLS PIPELINE II=1
		destination_buff[idx_offset + i] = source_buff[source_offset + i];
	}
}


template<typename T1, typename T2, len_t dest_dim_0, len_t dest_dim_1>
void copy_to_buff_par(
	const T1 *source_buff,
	T2 destination_buff[dest_dim_0][dest_dim_1],
	const len_t source_offset,
	const len_t dim_0,
	const len_t dim_1
	) {

	#pragma HLS INLINE

	const len_t rounded_up_dim_1 = DIV_ROUNDUP(dim_1, dest_dim_1) * dest_dim_1;
	// if (dim_1 != rounded_up_dim_1) {
	// 	std::cout << dim_0 << " " << dim_1 << "\n";
	// 	std::cout << "IMPERFECT ARRAY: " << dim_1 << " => " << rounded_up_dim_1 << "\n";
	// }

	const len_t n_vecs = (dim_0 * rounded_up_dim_1) / PORT_VEC_SIZE;
	const len_t vec_offset = source_offset / PORT_VEC_SIZE;

	copy_to_buff_par_lp:
	for (len_t i = 0; i < n_vecs; i++) {
		#pragma HLS loop_tripcount min=BUF_CP_PAR_MIN_CNT max=BUF_CP_PAR_MAX_CNT
		#pragma HLS PIPELINE II=1

		T1 temp_vec = source_buff[vec_offset + i];
		for (uint8_t j = 0; j < PORT_VEC_SIZE; j++) {
			#pragma HLS UNROLL
			destination_buff[i][j] = temp_vec[j];
		}
	}

	// if (dim_1 != rounded_up_dim_1) {
	// 	for (len_t i = 0; i < n_vecs; i++) {
	// 		std::cout << i << ": \t";
	// 		for (len_t j = 0; j < PORT_VEC_SIZE; j++) {
	// 			std::cout << destination_buff[i][j] << " ";
	// 		}
	// 		std::cout << "\n";
	// 	}
	// }
}


template<typename T1, typename T2, len_t s_dim_0, len_t s_dim_1, len_t d_dim_0, len_t d_dim_1>
void copy_from_to_buff_par(
	const T1 source_buff[s_dim_0][s_dim_1],
	T2 destination_buff[d_dim_0][d_dim_1],
	const len_t len
	) {

	#pragma HLS INLINE

	const len_t n_vecs = len / TILE_SIZE;

	copy_from_to_buff_par_lp:
	for (len_t i = 0; i < n_vecs; i++) {
		#pragma HLS loop_tripcount min=BUF_CP_PAR_MIN_CNT max=BUF_CP_PAR_MAX_CNT
		#pragma HLS PIPELINE II=1

		for (uint8_t j = 0; j < TILE_SIZE; j++) {
			#pragma HLS UNROLL
			destination_buff[i][j] = source_buff[i][j];
		}
	}
}


template<typename T1, typename T2, len_t source_dim_0, len_t source_dim_1>
void write_from_buff_par(
	const T1 source_buff[source_dim_0][source_dim_1],
	T2 *destination_buff,
	const len_t destination_offset,
	const len_t dim_0,
	const len_t dim_1
	) {

	#pragma HLS INLINE

	const len_t rounded_up_dim_1 = DIV_ROUNDUP(dim_1, source_dim_1) * source_dim_1;
	const len_t n_vecs = (dim_0 * rounded_up_dim_1) / PORT_VEC_SIZE;
	const len_t vec_offset = destination_offset / PORT_VEC_SIZE;

	write_from_buff_par_lp:
	for (len_t i = 0; i < n_vecs; i++) {
		#pragma HLS loop_tripcount min=(RESMLP_OUT_PADDED_DIM_0*RESMLP_OUT_PADDED_DIM_1) max=(RESMLP_OUT_PADDED_DIM_0*RESMLP_OUT_PADDED_DIM_1)
		#pragma HLS PIPELINE II=1

		T2 temp_vec;
		for (uint8_t j = 0; j < PORT_VEC_SIZE; j++) {
			#pragma HLS UNROLL
			temp_vec[j] = source_buff[i][j];
		}

		destination_buff[vec_offset + i] = temp_vec;
	}
}


#endif // DMA_UTILS_H
