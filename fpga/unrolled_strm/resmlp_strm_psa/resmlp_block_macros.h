#ifndef RESMLP_RELU_H_MUL_1_BLOCKS_H
#define RESMLP_RELU_H_MUL_1_BLOCKS_H


#include "resmlp_token_mixer.h"
#include "resmlp_mlp.h"


// Helper macro to resmlp_token_mixer with correct inputs
#define CALL_RESMLP_TOKEN_MIXER(n, in, out) \
resmlp_token_mixer( \
	in, \
	blocks_##n##_token_mixer_affine_1_w, \
	blocks_##n##_token_mixer_affine_1_b, \
	blocks_##n##_token_mixer_affine_1_s, \
	blocks_##n##_token_mixer_linear_w, \
	blocks_##n##_token_mixer_linear_b, \
	blocks_##n##_token_mixer_linear_s, \
	blocks_##n##_token_mixer_affine_2_w, \
	blocks_##n##_token_mixer_affine_2_b, \
	blocks_##n##_token_mixer_affine_2_s, \
	blocks_##n##_token_mixer_res_affine_w, \
	blocks_##n##_token_mixer_res_affine_s, \
	out);


#define CALL_RESMLP_CROSS_CHANNEL(n, in, out) \
resmlp_mlp( \
	in, \
	blocks_##n##_mlp_psa_1_lerp_lut, \
	blocks_##n##_mlp_linear_1_w, \
	blocks_##n##_mlp_linear_1_b, \
	blocks_##n##_mlp_linear_1_s, \
	blocks_##n##_mlp_psa_2_lerp_lut, \
	blocks_##n##_mlp_linear_2_w, \
	blocks_##n##_mlp_linear_2_b, \
	blocks_##n##_mlp_linear_2_s, \
	blocks_##n##_mlp_res_affine_w, \
	blocks_##n##_mlp_res_affine_s, \
	out);


#endif //RESMLP_RELU_H_MUL_1_BLOCKS_H
