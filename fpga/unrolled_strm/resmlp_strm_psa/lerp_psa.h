#ifndef LERP_PSA_H
#define LERP_PSA_H


#include "resmlp_utils.h"
#include "resmlp_types.h"
#include <ap_int.h>


typedef ap_int<8> model_psa_t;
typedef ap_int<9> model_psa_off_t;
typedef ap_uint<8> model_psa_idx_t;


#define PSA_IDX_OFFSET 128

#define LERP_PSA_BITS 8
#define LERP_PSA_MSB_BITS 6
#define LERP_PSA_LSB_BITS (LERP_PSA_BITS - LERP_PSA_MSB_BITS)
#define LERP_PSA_N (1<<LERP_PSA_MSB_BITS)

// #define LERP_PSA_LUT_SIZE LERP_PSA_N + 1
#define LERP_PSA_LUT_SIZE LERP_PSA_N

typedef ap_uint<LERP_PSA_MSB_BITS> psa_msb_idx_t;
typedef ap_uint<LERP_PSA_LSB_BITS> psa_lsb_idx_t;

typedef ap_int<LERP_PSA_BITS*2> psa_lerp_t;


model_act_t lerp_psa_lookup(
	model_act_t x,
	const model_psa_t psa_lut[LERP_PSA_LUT_SIZE]
	) {

	#pragma HLS INLINE

	model_psa_off_t tmpx = x;
	model_psa_idx_t x_idx = (tmpx + (model_psa_off_t) PSA_IDX_OFFSET);

	// Split index into MSB and LSB portion for LERP
	psa_msb_idx_t x_msb = x_idx.range(LERP_PSA_BITS-1, LERP_PSA_LSB_BITS);
	psa_msb_idx_t x_lsb = x_idx.range(LERP_PSA_LSB_BITS-1, 0);

	// Clip the x_msb to be within the range
	// CLIP_MAX(x_msb, ((psa_msb_idx_t)(LERP_PSA_N - 2)));

	model_psa_t y0 = psa_lut[x_msb];
	model_psa_t y1 = psa_lut[x_msb+1];

	psa_lerp_t y = (y1 - y0) * x_lsb;
	#pragma HLS BIND_OP variable=y op=mul impl=fabric

	// y >>= LERP_PSA_LSB_BITS;
	ROUNDED_SHIFT(y, LERP_PSA_LSB_BITS);

	y += y0;

	model_act_t out_y = (model_act_t) y;
	return out_y;
}


template<unsigned int dim0, unsigned int dim1>
void compute_lerp_psa(
	hls::stream<model_act_t> &strm_in,
	hls::stream<model_act_t> &strm_out,
	const model_psa_t psa_lut[dim1][LERP_PSA_LUT_SIZE]
	) {

	#pragma HLS INLINE off
	#pragma HLS bind_storage variable=psa_lut type=ROM_2P impl=BRAM

	for (int i = 0; i < dim0; i++) {
		for (int j = 0; j < dim1; j++) {
        #pragma HLS PIPELINE II=1

			model_act_t x;
			strm_in >> x;
			model_act_t y = lerp_psa_lookup(x, psa_lut[j]);
			strm_out << y;
		}
	}
}


#endif // LERP_PSA_H
