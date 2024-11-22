#ifndef RESMLP_H
#define RESMLP_H

#include <stdint.h>
#include "accel_common.h"


void krnl_resmlp(
	// Input
	const accel_io_vec_t *x_vec,
	// FC Layer Params
	const accel_w_vec_t *w_vec,
	const accel_b_vec_t *b_vec,
	const accel_s_vec_t *s_vec,
	// Affine Layer Params
	const accel_w_vec_t *w_aff_vec,
	const accel_b_vec_t *b_aff_vec,
	const accel_s_vec_t *s_aff_vec,
	// Output
	accel_io_vec_t *y_vec
);


#endif // RESMLP_H
