#include <iostream>
#include <math.h>

#include "resmlp.h"


// #include "quant_samples_h_mul_1.h"
// #include "quant_samples_extra_h_mul_1.h"

#include "quant_samples.h"
#include "quant_samples_extra.h"

#include <hls_stream.h>

#define BATCH_SIZE 8
#define MAX_ERR_TOL 0

#define SAMPLE_IN_VAR samples_input

// #define SAMPLE_OUT_VAR samples_inter_0
// #define SAMPLE_OUT_VAR samples_output

// #define SAMPLE_OUT_VAR(b, i, j) samples_inter_0[b][i][j]
// #define SAMPLE_OUT_VAR(b, i, j) samples_token_mixer_affine_1[i][j]
// #define SAMPLE_OUT_VAR(b, i, j) samples_token_mixer_linear_1[i][j]
// #define SAMPLE_OUT_VAR(b, i, j) samples_token_mixer_affine_2[i][j]
// #define SAMPLE_OUT_VAR(b, i, j) samples_token_mixer_res[i][j]
// #define SAMPLE_OUT_VAR(b, i, j) samples_token_mixer_post[i][j]
// #define SAMPLE_OUT_VAR(b, i, j) samples_inter_1[b][i][j]
// #define SAMPLE_OUT_VAR(b, i, j) samples_inter_2[b][i][j]
// #define SAMPLE_OUT_VAR(b, i, j) samples_inter_3[b][i][j]
// #define SAMPLE_OUT_VAR(b, i, j) samples_inter_4[b][i][j]
#define SAMPLE_OUT_VAR(b, i, j) samples_output[b % 8][j]


int main () {

	model_act_t hardware_output[BATCH_SIZE][OUT_SEQ_N][OUT_VEC_N];

	hls::stream<model_act_t> in;
	hls::stream<model_act_t> out;

	for (int b = 0; b < BATCH_SIZE; b++) {
		for (int i = 0; i < IN_SEQ_N; i++) {
			for (int j = 0; j < IN_VEC_N; j++) {
				model_act_t x = SAMPLE_IN_VAR[b % 8][i][j];
				in << x;
			}
		}

		krnl_resmlp(in, out);

		for (int i = 0; i < OUT_SEQ_N; i++) {
			for (int j = 0; j < OUT_VEC_N; j++) {
				out >> hardware_output[b][i][j];
			}
		}
	}

	// for (int b = 0; b < BATCH_SIZE; b++) {
	// 	krnl_resmlp(in, out);
	// }

	// for (int b = 0; b < BATCH_SIZE; b++) {
	// 	for (int i = 0; i < OUT_SEQ_N; i++) {
	// 		for (int j = 0; j < OUT_VEC_N; j++) {
	// 			out >> hardware_output[b][i][j];
	// 		}
	// 	}
	// }

	float err_sum = 0;
	int max_err_idx = -1;
	int max_err_bidx = -1;
	float max_err = 0;
	for (int b = 0; b < BATCH_SIZE; b++) {
		for (int i = 0; i < OUT_SEQ_N; i++) {
			for (int j = 0; j < OUT_VEC_N; j++) {
				float err = std::fabs((float)hardware_output[b][i][j] - (float)SAMPLE_OUT_VAR(b,i,j));
				if (err > max_err) {
					max_err = err;
					max_err_idx = i;
					max_err_bidx = b;
				}
				err_sum += err;
			}
		}
	}

//	max_err_idx = 0;
	if (max_err_idx == -1 || max_err <= MAX_ERR_TOL) {
		std::cout << "PASSED" << std::endl;
	} else{
		std::cout << "Avg Abs Err: " << err_sum / (BATCH_SIZE * OUT_SEQ_N * OUT_VEC_N) << std::endl;
		std::cout << "HARDWARE\t\tREFERENCE\t" << "idx: " << max_err_idx << std::endl;
		for (int j = 0; j < OUT_VEC_N; j++) {
			std::cout << (int)hardware_output[max_err_bidx][max_err_idx][j] << "\t\t <--> \t\t"
					<< (int)SAMPLE_OUT_VAR(max_err_bidx, max_err_idx, j) << std::endl;
		}

		// for (int b = 0; b < BATCH_SIZE; b++) {
		// 	for (int i = 0; i < OUT_SEQ_N; i++) {
		// 		for (int j = 0; j < OUT_VEC_N; j++) {
		// 			float err = std::fabs((float)hardware_output[b][i][j] - (float)SAMPLE_OUT_VAR(b,i,j));
		// 			if (err > 0) {
		// 				std::cout << std::endl << b << " " << i << " " << j << std::endl;
		// 				std::cout << (int) hardware_output[b][i][j] << " " << (int) SAMPLE_OUT_VAR(b,i,j) << std::endl;
		// 			}
		// 		}
		// 	}
		// }
	}

	return 0;
}
