#ifndef TB_HELPER_H
#define TB_HELPER_H

#include <random>
#include "utils.h"
#include "math.h"


/* generate a random value weighted within the normal (gaussian) distribution */
double gauss() {
	static std::default_random_engine generator (0);
	static std::normal_distribution<double> distribution (0.0, 1.0);
	return distribution(generator);
}


template<typename T, uint32_t dim_0>
void random_init_vec(
	T x[dim_0],
	double mu,
	double sigma_ratio,
	bool apply_abs
	) {

	for (uint32_t d0 = 0; d0 < dim_0; d0++) {
		double r = gauss();
		r = (r - mu) * sigma_ratio;
		r = apply_abs ? std::fabs(r) : r;

		x[d0] = ((int32_t) r) % 128;
	}
}

template<typename T, uint32_t dim_0, uint32_t dim_1>
void random_init_mat(
	T x[dim_0 * dim_1],
	double mu,
	double sigma_ratio,
	bool apply_abs
	) {


	for (uint32_t d0 = 0; d0 < dim_0; d0++) {
		for (uint32_t d1 = 0; d1 < dim_1; d1++) {

			double r = gauss();
			r = (r - mu) * sigma_ratio;
			r = apply_abs ? std::fabs(r) : r;

			int idx = IDX2D(d0, d1, dim_1);
			x[idx] = ((int32_t) r) % 128;
		}
	}
}


template<typename T>
void copy(const T *source, T *dest, const size_t len) {
	for (uint32_t i = 0; i < len; i++) {
		dest[i] = source[i];
	}
}


template<typename T1, typename T2, uint32_t vec_size>
void copy_to_vec(const T1 *source, T2 *dest, const size_t len) {
	const uint32_t n_vecs = ceil(len / (double) vec_size);
	for (uint32_t i = 0; i < n_vecs; i++) {
		for (uint32_t j = 0; j < vec_size; j++) {

			// Bound check to add zero padding
			if ((i * vec_size) + j < len)
				dest[i][j] = source[(i * vec_size) + j];
			else
				dest[i][j] = 0;
		}
	}
}


template<typename T1, typename T2, uint32_t vec_size>
void copy_to_vec(const T1 *source, T2 *dest, const size_t dim_0, const size_t dim_1) {
	const size_t rounded_up_dim_1 = DIV_ROUNDUP(dim_1, vec_size) * vec_size;
	if (rounded_up_dim_1 != dim_1) {
		std::cout << dim_0 << " " << dim_1 << "\n";
		std::cout << "IMPERFECT ARRAY: " << dim_1 << " => " << rounded_up_dim_1 << "\n";
	}

	const uint32_t n_vecs = ceil((dim_0 * rounded_up_dim_1) / (double) vec_size);

	uint32_t source_idx = 0;

	for (uint32_t i = 0; i < n_vecs; i++) {
		for (uint32_t j = 0; j < vec_size; j++) {

			// Bound check to add zero padding
			if (j < dim_1) {
				dest[i][j] = source[source_idx++];
			} else {
				dest[i][j] = 0;
			}
		}
	}

	// if (rounded_up_dim_1 != dim_1) {
	// 	for (uint32_t i = 0; i < n_vecs; i++) {
	// 		std::cout << i << ": \t";
	// 		for (uint32_t j = 0; j < vec_size; j++) {
	// 			std::cout << dest[i][j] << " ";
	// 		}
	// 		std::cout << "\n";
	// 	}
	// }
}


template<typename T1, typename T2, uint32_t vec_size>
void set_vecs(T1 *source, const T2 value, const size_t len) {
	const uint32_t n_vecs = ceil(len / (double) vec_size);
	for (uint32_t i = 0; i < n_vecs; i++) {
		for (uint32_t j = 0; j < vec_size; j++) {

			// Bound check to ensure correct read
			if ((i * vec_size) + j < len)
				source[i][j] = value;
			else
				source[i][j] = 0;
		}
	}
}


template<typename T1, uint32_t vec_size>
void split_4(
	const T1 *source,
	T1 *dest1,
	T1 *dest2,
	T1 *dest3,
	T1 *dest4,
	const size_t len
	) {

	const uint32_t n_vecs = ceil(len / (vec_size * 4.0));
	for (uint32_t i = 0; i < n_vecs; i++) {
		for (uint32_t j = 0; j < vec_size; j++) {

			// Bound check to ensure correct read
			int idx1 = (i * vec_size) + j;
			int idx2 = ((i+1) * vec_size) + j;
			int idx3 = ((i+2) * vec_size) + j;
			int idx4 = ((i+3) * vec_size) + j;

			if (idx1 < len)
				dest1[i][j] = source[i][j];
			else
				dest1[i][j] = 0;

			if (idx2 < len)
				dest2[i][j] = source[i+1][j];
			else
				dest2[i][j] = 0;

			if (idx3 < len)
				dest3[i][j] = source[i+2][j];
			else
				dest3[i][j] = 0;

			if (idx4 < len)
				dest4[i][j] = source[i+3][j];
			else
				dest4[i][j] = 0;
		}
	}
}


#endif // TB_HELPER_H
