#ifndef ACCEL_COMMON_H
#define ACCEL_COMMON_H

#include <stdint.h>
#include <ap_int.h>
#include <hls_vector.h>
// #include <ap_fixed.h>

#define TILE_SIZE 16

#define MAX_B_DIM_LG2 6
#define MAX_B_DIM 64

// #define MAX_I_DIM 96
#define MAX_I_DIM 192

// Default datatype used for all dimensions and index calculations
typedef int16_t dim_t;
// typedef int32_t dim_t;

// Datatype used to store offsets and length of arrays in host/device memory
typedef int32_t len_t;

// Default datatype of input/output
// Used for activations, and layer output post clip
typedef ap_int<8> accel_io_t;
typedef ap_int<9> accel_res_add_t;

#define ACCEL_IO_CLIP_MAX 127
#define ACCEL_IO_CLIP_MIN -128

// Datatype for parameters
typedef ap_int<8> accel_w_t;	// Weight
typedef ap_int<8> accel_b_t;	// Bias
typedef ap_uint<8> accel_s_t; 	// Scale Shift

// Datatype used for accumulation and multiplication output
typedef ap_int<18> accel_mul_t;
// typedef ap_int<22> accel_acc_t;     // ceil(log2(128x128x96)) + 1
typedef ap_int<23> accel_acc_t;     // ceil(log2(128x128x192)) + 1

// Should be equal to TILE_SIZE
// o.w. code needs modification
#define PORT_VEC_SIZE TILE_SIZE

// Data type used at port-level of kernel
typedef hls::vector<accel_io_t, PORT_VEC_SIZE> accel_io_vec_t;
typedef hls::vector<accel_w_t, PORT_VEC_SIZE> accel_w_vec_t;
typedef hls::vector<accel_b_t, PORT_VEC_SIZE> accel_b_vec_t;
typedef hls::vector<accel_s_t, PORT_VEC_SIZE> accel_s_vec_t;

// Sizes and number of buffers
#define MAX_IO_BUFFER (MAX_B_DIM * MAX_I_DIM)
#define IO_BUFF_DIM_0 ((MAX_IO_BUFFER)/(TILE_SIZE))
#define IO_BUFF_DIM_1 (TILE_SIZE)
#define IO_BUFF_N 3

#define MAX_W_BUFFER (MAX_I_DIM * MAX_I_DIM)
#define W_BUFF_DIM_0 ((MAX_W_BUFFER)/(TILE_SIZE))
#define W_BUFF_DIM_1 (TILE_SIZE)

#define MAX_BS_BUFFER MAX_I_DIM
#define BS_BUFF_DIM_0 ((MAX_BS_BUFFER)/(TILE_SIZE))
#define BS_BUFF_DIM_1 (TILE_SIZE)

#define WBS_BUFF_N 2

#define MAX_AFF_W_BUFFER MAX_I_DIM
#define W_AFF_BUFF_DIM_0 ((MAX_AFF_W_BUFFER)/(TILE_SIZE))
#define W_AFF_BUFF_DIM_1 (TILE_SIZE)

#define MAX_AFF_BS_BUFFER MAX_I_DIM
#define BS_AFF_BUFF_DIM_0 ((MAX_BS_BUFFER)/(TILE_SIZE))
#define BS_AFF_BUFF_DIM_1 (TILE_SIZE)

#define WBS_AFF_BUFF_N 6

#endif // ACCEL_COMMON_H
