import numpy as np
import pickle
from np_to_h import (h_header, h_footer, ndarray_to_var, var_to_var)
from huffman import (huffman_compress, print_huffman_tree)
import pprint
import math

from diff_compression import diff_based_luts
from psa_lut_compress import compress_psa_luts
from pwl_compress import pwl_compress
from lerp_compress import lerp_compress
from delta_compression import delta_based_luts


def gen_split_lut(param_dict):
    for i in range(len(param_dict['blocks'])):
        k = 'mlp'
        for k1, v1 in param_dict['blocks'][i][k].items():
            if 'psa' in k1:
                lut = v1['lut']
                split_idx = lut.shape[-1]//2
                neg_lut = lut[:, :split_idx]
                pos_lut = lut[:, split_idx:]
                v1['neg_lut'] = neg_lut
                v1['pos_lut'] = pos_lut
                v1.pop('lut')


def gen_diff_based_encoded(param_dict, tile_factor=8, compression_type=1, pop_og_lut=False):
    final_uncompressed_size = 0
    final_compressed_size = 0

    for i in range(len(param_dict['blocks'])):
        k = 'mlp'
        for k1, v1 in param_dict['blocks'][i][k].items():
            if 'psa' in k1:
                lut = v1['lut']
                total_compressed_size, extra_luts = diff_based_luts(
                    lut, tile_factor, compression_type=compression_type)
                if pop_og_lut:
                    v1.pop('lut')
                v1.update(extra_luts)

                uncompressed_size = lut.size * 8
                final_uncompressed_size += uncompressed_size
                final_compressed_size += total_compressed_size

    return final_uncompressed_size, final_compressed_size


def gen_ternary_encoded(param_dict, target_bits=None):
    for i in range(len(param_dict['blocks'])):
        k = 'mlp'
        for k1, v1 in param_dict['blocks'][i][k].items():
            if 'psa' in k1:
                compress_psa_luts(v1, max_target_bits=target_bits)
                # unq = np.unique(v1['delta'])
                # unq_len = unq.size
                # v1['delta_bits'] = int(np.floor(np.log2(unq_len)) + 1)
                print(v1['delta_bits'])


                # with np.nditer(v1['delta'], op_flags=['readwrite']) as it:
                #     for x in it:
                #         x =

                v1.pop('lut')
                # return


def gen_pwl_based_encoded(param_dict, segments=4):
    for i in range(len(param_dict['blocks'])):
        k = 'mlp'
        for k1, v1 in param_dict['blocks'][i][k].items():
            if 'psa' in k1:
                lut = v1['lut']
                pwl_compress(lut, segments)


def gen_lerp_based_encoded(param_dict, points=2**6, report_err=False, append=True, pop_og_lut=False, transpose=False, tiled=None):
    for i in range(len(param_dict['blocks'])):
        k = 'mlp'
        for k1, v1 in param_dict['blocks'][i][k].items():
            if 'psa' in k1:
                lut = v1['lut']
                lerp_lut = lerp_compress(lut, points, report_err=report_err, append=append)
                if transpose:
                    lerp_lut = lerp_lut.T
                if tiled:
                    lerp_lut = lerp_lut.reshape(lerp_lut.shape[0], -1, tiled).swapaxes(0,1)
                v1['lerp_lut'] = lerp_lut
                if pop_og_lut:
                    v1.pop('lut')


def gen_lerp_diff_based_encoded(param_dict, points=2**6, tile_factor=2, compression_type=1, pop_og_lut=False, append=True):
    final_uncompressed_size = 0
    final_compressed_size = 0

    for i in range(len(param_dict['blocks'])):
        k = 'mlp'
        for k1, v1 in param_dict['blocks'][i][k].items():
            if 'psa' in k1:
                lut = v1['lut']
                lerp_lut = lerp_compress(lut, points, append=append, report_err=False)
                total_compressed_size, extra_luts = diff_based_luts(
                    lerp_lut, tile_factor, compression_type=compression_type, append=append, pre_appended=append)
                if pop_og_lut:
                    v1.pop('lut')
                v1.update(extra_luts)

                uncompressed_size = lerp_lut.size * 8
                final_uncompressed_size += uncompressed_size
                final_compressed_size += total_compressed_size

    return final_uncompressed_size, final_compressed_size


def gen_lerp_delta_based_encoded(param_dict, points=2**6, tile_factor=2, compression_type=1, pop_og_lut=False, append=True):
    final_uncompressed_size = 0
    final_compressed_size = 0

    for i in range(len(param_dict['blocks'])):
        k = 'mlp'
        for k1, v1 in param_dict['blocks'][i][k].items():
            if 'psa' in k1:
                lut = v1['lut']
                lerp_lut = lerp_compress(lut, points, append=append, report_err=False)

                # delta_based_luts(
                #     lerp_lut, tile_factor, compression_type=compression_type,
                #     append=append, pre_appended=append)

                total_compressed_size, extra_luts = delta_based_luts(
                    lerp_lut, tile_factor, compression_type=compression_type, append=append, pre_appended=append)
                if pop_og_lut:
                    v1.pop('lut')
                v1.update(extra_luts)

                uncompressed_size = lerp_lut.size * 8
                final_uncompressed_size += uncompressed_size
                final_compressed_size += total_compressed_size

    return final_uncompressed_size, final_compressed_size


def write_psa_params_to_h(param_dict, param_h_file,
        var_type='ap_int<8>', includes=['#include <ap_int.h>']):

    def layer_to_s(param, prefix):
        s = ''
        for k, v in param.items():
            # Ignore anything that is not PSA
            if 'psa' not in prefix:
                continue

            var_type_tmp = var_type
            if 'psa' in prefix:
                if k == 'offset':
                    continue
                # Ternary Based Lossy Compression
                elif k == 'w':
                    var_type_tmp = 'ap_int<2>'
                elif k == 'delta_shft_amnt':
                    var_type_tmp = 'ap_uint<3>'
                elif (k == 'delta_shifted' or k == 'delta'):
                    new_type = prefix + '_' + k + '_t'
                    s += f'\ntypedef ap_int<{param["delta_bits"]}> {new_type};'
                    var_type_tmp = new_type
                elif k == 'delta_bits':
                    continue
                # Diff Based Com/pression
                elif k == 'lut_diff':
                    new_type = prefix + '_' + k + '_t'
                    s += f'\ntypedef ap_int<{param["lut_diff_bits"]}> {new_type};'
                    var_type_tmp = new_type
                elif k == 'lut_diff_bits':
                    continue
                elif k == 'l0_lut_diff':
                    new_type = prefix + '_' + k + '_t'
                    s += f'\ntypedef ap_int<{param["l0_lut_diff_bits"]}> {new_type};'
                    var_type_tmp = new_type
                elif k == 'l0_lut_diff_bits':
                    continue
                elif k == 'l1_lut_diff':
                    new_type = prefix + '_' + k + '_t'
                    s += f'\ntypedef ap_int<{param["l1_lut_diff_bits"]}> {new_type};'
                    var_type_tmp = new_type
                elif k == 'l1_lut_diff_bits':
                    continue
                elif k == 'feat_using_l1':
                    new_type = prefix + '_' + k + '_t'
                    s += f'\ntypedef ap_uint<1> {new_type};'
                    var_type_tmp = new_type
                elif k == 'feat_translation':
                    new_var = prefix + '_' + k
                    s += f'\nconst unsigned int {new_var}_count = {v.shape[0]};'
                    var_type_tmp = 'ap_uint<8>'

                elif k == 'idx_lookup':
                    new_type = prefix + '_' + k + '_t'
                    s += f'\ntypedef ap_uint<{param["idx_lookup_bits"]}> {new_type};'
                    var_type_tmp = new_type
                elif k == 'idx_lookup_bits':
                    continue

                elif k == 'unq_diff':
                    new_type = prefix + '_' + k + '_t'
                    s += f'\ntypedef ap_int<{param["unq_diff_bits"]}> {new_type};'
                    var_type_tmp = new_type
                elif k == 'unq_diff_bits':
                    continue

                # LERP Approximation
                elif k == 'lerp_lut':
                    pass
                elif v is None:
                    continue


            print(prefix, k, var_type_tmp)
            if type(v) == np.array or type(v) == np.ndarray:
                s += ndarray_to_var(v, f'const {var_type_tmp}', prefix + '_' + k)
            else:
                s += var_to_var(v, f'const {var_type_tmp}', prefix + '_' + k)
        return s



    with open(param_h_file, 'w') as f:
        f.write(h_header(param_h_file, includes=includes))

        for i in range(len(param_dict['blocks'])):
            k = 'mlp'
            prefix = f'blocks_{i}_{k}'
            for k2, v2 in param_dict['blocks'][i][k].items():
                if k2 == 'psa_1' or k2 == 'psa_2':
                    f.write(layer_to_s(v2, prefix + '_' + k2))

        f.write(h_footer(param_h_file))


if __name__ == "__main__":
    # parent_folder = 'hardware/resmlp_layers_4_embed_96_h_mul_1/baseline'
    # parent_folder = 'hardware/resmlp_layers_4_embed_96_h_div_2/psa'
    parent_folder = 'hardware/lerp_psa/resmlp_layers_4_embed_96_h_mul_1/lerp_psa_8_64_epsilon_1e-0'
    # parent_folder = 'hardware/lerp_psa/resmlp_layers_4_embed_96_h_div_2/lerp_psa_8_128_epsilon_1e-0'
    # parent_folder = './'
    params_pkl_file = f'{parent_folder}/quant_params.pkl'

    with open(params_pkl_file, 'rb') as fp:
        param_dict = pickle.load(fp)

        # gen_split_lut(param_dict)

        # gen_huffman_encoded(param_dict)

        # for tile_factor in [1, 2, 4, 8, 16, 32, 64, 128]:
        # for tile_factor in [1, 2, 4, 8, 16, 32]:
        #     unc_s, cmp_s = gen_diff_based_encoded(param_dict, tile_factor=tile_factor, compression_type=8)
        #     print(f'[{tile_factor}] => ({unc_s}, {cmp_s}) - {unc_s/cmp_s:.2f}')

        # gen_diff_based_encoded(param_dict, tile_factor=4, compression_type=1)

        # gen_pwl_based_encoded(param_dict, segments=64)

        # gen_lerp_based_encoded(param_dict, points=2**6, report_err=True, pop_og_lut=True, append=False, transpose=True, tiled=16)
        # gen_lerp_based_encoded(param_dict, points=2**6, report_err=True, pop_og_lut=True, append=False)

        # gen_ternary_encoded(param_dict, target_bits=6)

        # for tile_factor in [1, 2, 4, 8, 16, 32]:
        # for tile_factor in [16]:
            # unc_s, cmp_s = gen_lerp_diff_based_encoded(param_dict, points=2**6, tile_factor=tile_factor, compression_type=8)
            # print(f'[{tile_factor}] => ({unc_s}, {cmp_s}) - {unc_s/cmp_s:.2f}')
        gen_lerp_diff_based_encoded(param_dict, points=2**6, tile_factor=8, compression_type=1, pop_og_lut=True, append=True)

        # for tile_factor in [1, 2, 4, 8, 16, 32]:
        # # for tile_factor in [16]:
        #     unc_s, cmp_s = gen_lerp_delta_based_encoded(param_dict, points=2**6, tile_factor=tile_factor)
        #     print(f'[{tile_factor}] => ({unc_s}, {cmp_s}) - {unc_s/cmp_s:.2f}')

        # print(param_dict)
        # write_psa_params_to_h(param_dict, f'{parent_folder}/lerp_psa_params_transposed.h')
        write_psa_params_to_h(param_dict, f'{parent_folder}/lerp_diff_psa_params.h')
