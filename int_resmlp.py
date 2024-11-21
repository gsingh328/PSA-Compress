import numpy as np
import pickle
import numpy as np
import math

from np_to_h import (h_header, h_footer, ndarray_to_var, var_to_var)

from tqdm import tqdm
import timeit


ACT_TYPE = np.int8      # used for truncation error
ACT_BITS = 8
ACT_S = 2**(ACT_BITS-1)
ACT_MIN = -2**(ACT_BITS-1)
ACT_MAX = 2**(ACT_BITS-1) - 1

WGT_BITS = 8
WGT_S = 2**(WGT_BITS-1)
WGT_MIN = -2**(WGT_BITS-1)
WGT_MAX = 2**(WGT_BITS-1) - 1

LERP_DROP_LAST = False


def int_linear(x, w, b, s, use_rounded_shift=True, clip_output=True):
    assert x.shape[1] == w.shape[1]
    assert w.shape[0] == b.shape[0]
    assert len(s) == 1 or s.shape[0] == w.shape[0]

    y = np.matmul(x, w.T)

    if use_rounded_shift:
        y = y >> (s - 1)
        y_rnd_factor = y & 1
        y = (y >> 1) + y_rnd_factor
    else:
        y = y >> s

    if b is not None:
        y = y + b

    if clip_output:
        y = np.clip(y, ACT_MIN, ACT_MAX)
    else:
        # cast to correct type and then upcast.
        # properly propogates the truncation error
        y = y.astype(ACT_TYPE).astype(np.int32)
    return y


def float_linear(x, w, b):
    y = np.matmul(x, w.T)
    if b is not None:
        y = y + b
    return y


def int_affine(x, w, b, s, use_rounded_shift=True, clip_output=True):
    assert x.shape[1] == w.shape[0], f'{x.shape} <-/-> {w.shape}'
    assert w.shape[0] == b.shape[0], f'{w.shape} <-/-> {b.shape}'
    assert len(s) == 1 or s.shape[0] == w.shape[0]

    y = x * w

    if use_rounded_shift:
        y = y >> (s - 1)
        y_rnd_factor = y & 1
        y = (y >> 1) + y_rnd_factor
    else:
        y = y >> s

    if b is not None:
        y = y + b

    if clip_output:
        y = np.clip(y, ACT_MIN, ACT_MAX)
    else:
        # cast to correct type and then upcast.
        # properly propogates the truncation error
        y = y.astype(ACT_TYPE).astype(np.int32)
    return y


def float_affine(x, w, b):
    y =  x * w
    if b is not None:
        y = y + b
    return y


def linear(x, params:dict, op_type):
    assert 'w' in params
    w = params['w']
    b = params['b'] if 'b' in params else None
    if op_type == 'float':
        return float_linear(x, w, b)
    elif op_type == 'int':
        assert 's' in params
        s = params['s']
        return int_linear(x, w, b, s)
    else:
        raise ValueError(f"unknown op_type: {op_type}")


def affine(x, params:dict, op_type):
    assert 'w' in params
    w = params['w']
    b = params['b'] if 'b' in params else None
    if op_type == 'float':
        return float_affine(x, w, b)
    elif op_type == 'int':
        assert 's' in params
        s = params['s']
        return int_affine(x, w, b, s)
    else:
        raise ValueError(f"unknown op_type: {op_type}")


def add(a, b, op_type, clip_output=True):
    if op_type == 'float':
        return a + b
    elif op_type == 'int':
        y = a + b
        if clip_output:
            y = np.clip(y, ACT_MIN, ACT_MAX)
        else:
            # cast to correct type and then upcast.
            # properly propogates the truncation error
            y = y.astype(ACT_TYPE).astype(np.int32)
        return y
    else:
        raise ValueError(f"unknown op_type: {op_type}")


def psa(x, params:dict, compression_type=None):
    if compression_type is None:
        assert 'lut' in params
        assert 'offset' in params

        lut = params['lut']
        offset = params['offset']

        # assert lut.shape[1] == x.shape[1], "PSA is mismatched with input features"
        assert lut.shape[0] == x.shape[1], "PSA is mismatched with input features"

        y = np.zeros_like(x)
        for i in range(lut.shape[0]):
            # lut_i = lut[:, i]
            lut_i = lut[i, :]
            idx = x[:, i] + offset
            y[:, i] = lut_i[idx]
        return y

    if compression_type == 'lossy_ternary':
        assert 'breakpoint' in params
        assert 'w' in params
        assert 'b' in params
        assert 'delta_shft_amnt' in params
        assert 'delta_shifted' in params

        lut = params['delta_shifted']
        offset = params['offset']
        breakpoint = params['breakpoint']
        shft_amnt = params['delta_shft_amnt']
        w = params['w']
        b = params['b']
        y = np.zeros_like(x)
        for i in range(lut.shape[0]):
            lut_i = lut[i, :]
            x_i = x[:, i]
            bp_i = breakpoint[i]
            w_i = w[i, :]
            b_i = b[i, :]

            w_i = w_i[(x_i >= bp_i).astype(np.int32)]
            b_i = b_i[(x_i >= bp_i).astype(np.int32)]

            idx = x_i + offset
            y[:, i] = (w_i * x[:, i]) + b_i + (lut_i[idx] << shft_amnt)
        return y

    if compression_type == 'lerp':
        assert 'lerp_lut' in params
        assert 'offset' in params

        lerp_lut = params['lerp_lut']
        offset = params['offset']
        assert lerp_lut.shape[0] == x.shape[1], "LERP PSA is mismatched with input features"

        if LERP_DROP_LAST:
            lerp_lut = lerp_lut[..., :-1]
            points = lerp_lut.shape[-1]
        else:
            points = lerp_lut.shape[-1] - 1

        mbs_bits = int(math.ceil(math.log2(points)))
        lsb_bits = ACT_BITS - mbs_bits

        idx = x + offset
        x_msb = idx >> lsb_bits
        x_msb = np.clip(x_msb, 0, lerp_lut.shape[-1] - 2)
        x_lsb = idx & (2**lsb_bits - 1)

        y = np.zeros_like(x)
        for i in range(lerp_lut.shape[0]):
            lut_i = lerp_lut[i, :]
            x_msb_i = x_msb[:, i]
            x_lsb_i = x_lsb[:, i]

            y0 = lut_i[..., x_msb_i]
            y1 = lut_i[..., x_msb_i + 1]

            # Floored shifting
            # _y = ((y1 - y0) * x_lsb_i) >> lsb_bits

            # Rounded shifting
            _y = ((y1 - y0) * x_lsb_i) >> (lsb_bits-1)
            _y_rnd_factor = _y & 1
            _y = (_y >> 1) + _y_rnd_factor

            _y += y0

            y[:, i] = _y

        return y


def avgpool(x, axis):
    n = np.log2(x.shape[axis]).round().astype(np.int32)
    x = np.sum(x, axis, keepdims=True)
    x = x >> (n - 1)
    rnd_factor = x & 1
    x = (x >> 1) + rnd_factor
    return x


def resmlp_token_mixer(x, params:dict, op_type):
    i_shape = x.shape

    assert 'linear' in params
    assert 'res_affine' in params
    x_res = affine(x, params['res_affine'], op_type)

    assert 'affine_1' in params
    x = affine(x, params['affine_1'], op_type)

    # Note: Transpose since we are applying linear across the tokens
    x = x.T
    x = linear(x, params['linear'], op_type)
    x = x.T

    assert 'affine_2' in params
    x = affine(x, params['affine_2'], op_type)

    x = add(x, x_res, op_type)

    o_shape = x.shape
    assert i_shape == o_shape
    return x


def resmlp_mlp(x, params:dict, op_type, is_psa=False, act='relu', compression_type=None):
    i_shape = x.shape

    assert 'linear_1' in params
    assert 'linear_2' in params
    assert 'res_affine' in params
    if is_psa:
        assert op_type == 'int'
        assert 'psa_1' in params
        assert 'psa_2' in params

    x_res = affine(x, params['res_affine'], op_type)

    # First layer
    if is_psa:
        x = psa(x, params['psa_1'], compression_type=compression_type)
    x = linear(x, params['linear_1'], op_type)

    if act == 'relu':
        x[x < 0] = 0
    elif act == None:
        pass
    else:
        raise ValueError(f"unsupported act: {act}")

    # Second layer
    if is_psa:
        x = psa(x, params['psa_2'], compression_type=compression_type)
    x = linear(x, params['linear_2'], op_type)

    # Output
    x = add(x, x_res, op_type)

    o_shape = x.shape
    assert i_shape == o_shape
    return x


def resmlp_block(x, params:dict, op_type, is_psa=False, compression_type=None):
    i_shape = x.shape

    # Token Mixer
    assert 'token_mixer' in params
    x = resmlp_token_mixer(x, params['token_mixer'], op_type)

    # MLP
    assert 'mlp' in params
    act = 'relu' if not is_psa else None
    x = resmlp_mlp(x, params['mlp'], op_type, is_psa=is_psa, act=act, compression_type=compression_type)

    o_shape = x.shape
    assert i_shape == o_shape
    return x


def resmlp_classifier(x, params:dict, op_type):
    x = avgpool(x, 0)
    x = linear(x, params, op_type)
    return x


def resmlp(x, params:dict, op_type='float', is_psa=False, compression_type=None):
    # Embed Linear
    assert 'embed_linear' in params
    x = linear(x, params['embed_linear'], op_type)

    # Blocks
    assert 'blocks' in params
    assert type(params['blocks']) == list
    for i in range(len(params['blocks'])):
        x = resmlp_block(x, params['blocks'][i], op_type, is_psa=is_psa, compression_type=compression_type)

    # Classifiermlp_params
    assert 'classifier_linear' in params
    x = resmlp_classifier(x, params['classifier_linear'], op_type)

    return x


def test_int_linear():
    xf = np.random.randn(64, 96)
    wf = np.random.randn(48, 96)
    bf = np.random.randn(48)
    yf = np.matmul(xf, wf.T) + bf

    x_abs_max = np.max(np.abs(xf))
    x_s = ACT_S / x_abs_max
    xq = np.round(xf * x_s).clip(ACT_MIN, ACT_MAX).astype(np.int32)

    y_abs_max = np.max(np.abs(yf))
    y_s = ACT_S / y_abs_max
    yq = np.round(yf * y_s).clip(ACT_MIN, ACT_MAX).astype(np.int32)

    w_abs_max = np.max(np.abs(wf), axis=1).reshape(-1, 1)
    w_s = WGT_S / w_abs_max

    yq_s = (x_s * w_s) / y_s

    requant_s = (2.**(np.log2(yq_s).round()))
    w_s = (requant_s * y_s) / x_s
    wq = np.round(wf * w_s).clip(ACT_MIN, ACT_MAX).astype(np.int32)
    sq = np.round(np.log2(requant_s)).astype(np.int32).reshape(-1)
    bq = np.round(bf * y_s).clip(ACT_MIN, ACT_MAX).astype(np.int32)

    yqq = int_linear(xq, wq, bq, sq)

    err = yqq - yq
    print(np.mean(err), np.std(err), np.max(np.abs(err)))


def test_int_affine():
    xf = np.random.randn(64, 96)
    wf = np.random.randn(96)
    bf = np.random.randn(96)
    yf = xf * wf + bf

    x_abs_max = np.max(np.abs(xf))
    x_s = ACT_S / x_abs_max
    xq = np.round(xf * x_s).clip(ACT_MIN, ACT_MAX).astype(np.int32)

    y_abs_max = np.max(np.abs(yf))
    y_s = ACT_S / y_abs_max
    yq = np.round(yf * y_s).clip(ACT_MIN, ACT_MAX).astype(np.int32)

    w_abs_max = np.max(np.abs(wf))
    w_s = WGT_S / w_abs_max

    yq_s = (x_s * w_s) / y_s

    requant_s = (2.**(np.log2(yq_s).round()))
    w_s = (requant_s * y_s) / x_s
    wq = np.round(wf * w_s).clip(ACT_MIN, ACT_MAX).astype(np.int32)
    sq = np.round(np.log2(requant_s)).astype(np.int32).reshape(-1)
    bq = np.round(bf * y_s).clip(ACT_MIN, ACT_MAX).astype(np.int32)

    yqq = int_affine(xq, wq, bq, sq)

    err = yqq - yq
    print(np.mean(err), np.std(err), np.max(np.abs(err)))


def test_resmlp(nblocks=4):
    x = np.random.randn(64, 48)
    print(x.shape)
    params = {
        'embed_linear': {
            'w': np.random.randn(96, 48),
            'b': np.random.randn(96)
        },
        'blocks': [
            {
                'token_mixer': {
                    'affine_1': {
                        'w': np.random.randn(96),
                        'b': np.random.randn(96)
                    },
                    'linear': {
                        'w': np.random.randn(64, 64),
                        'b': np.random.randn(64)
                    },
                    'affine_2': {
                        'w': np.random.randn(96),
                        'b': np.random.randn(96)
                    },
                    'res_affine': {
                        'w': np.random.randn(96),
                        'b': np.random.randn(96)
                    }
                },

                'mlp': {
                    'linear_1': {
                        'w': np.random.randn(48, 96),
                        'b': np.random.randn(48)
                    },
                    'linear_2': {
                        'w': np.random.randn(96, 48),
                        'b': np.random.randn(96)
                    },
                    'res_affine': {
                        'w': np.random.randn(96),
                        'b': np.random.randn(96)
                    }
                }
            }
            for i in range(nblocks)
        ],
        'classifier_linear': {
            'w': np.random.randn(10, 96),
            'b': np.random.randn(10)
        }
    }
    y = resmlp(x, params)
    print(y.shape)


def test_resmlp_with_pkl(param_file, samples_file):
    with open(param_file, 'rb') as fp:
        param_dict = pickle.load(fp)
    with open(samples_file, 'rb') as fp:
        samples_dict = pickle.load(fp)

    errs = []
    for i in range(samples_dict['input'].shape[0]):
        x = samples_dict['input'][i]
        y = resmlp(x, param_dict)
        y_ref = samples_dict['output'][i]
        err = y - y_ref
        errs.append(err)

    errs = np.vstack(errs)
    print(errs.shape)
    print(np.mean(errs), np.std(errs), np.max(np.abs(errs)))


def test_quant_resmlp_with_pkl(param_file, samples_file):
    with open(param_file, 'rb') as fp:
        param_dict = pickle.load(fp)
    with open(samples_file, 'rb') as fp:
        samples_dict = pickle.load(fp)

    errs = []
    for i in range(samples_dict['input'].shape[0]):
        x = samples_dict['input'][i]
        y = resmlp(x, param_dict, 'int')
        y_ref = samples_dict['output'][i]
        err = y - y_ref
        errs.append(err)

    errs = np.vstack(errs)
    print(errs.shape)
    print(np.mean(errs), np.std(errs), np.max(np.abs(errs)))


def time_quant_resmlp_with_pkl(param_file, samples_file):
    with open(param_file, 'rb') as fp:
        param_dict = pickle.load(fp)
    with open(samples_file, 'rb') as fp:
        samples_dict = pickle.load(fp)

    def fn():
        for i in range(samples_dict['input'].shape[0]):
            x = samples_dict['input'][i]
            y = resmlp(x, param_dict, 'int')

    t = timeit.timeit(fn, number=10)
    print(t)


def test_all_quant_resmlp_with_pkl(param_file, samples_file):
    with open(param_file, 'rb') as fp:
        param_dict = pickle.load(fp)
    with open(samples_file, 'rb') as fp:
        samples_dict = pickle.load(fp)

    errs = []
    correct = 0
    total = 0
    for batch_idx in tqdm(range(len(samples_dict['input']))):
        for i in range(samples_dict['input'][batch_idx].shape[0]):
            x = samples_dict['input'][batch_idx][i]
            y = resmlp(x, param_dict, 'int')

            predicted = y.argmax(1)
            total += len(predicted)
            correct += int((predicted == samples_dict['labels'][batch_idx][i]).sum())

            y_ref = samples_dict['output'][batch_idx][i]
            err = y - y_ref
            errs.append(err)

    errs = np.vstack(errs)
    print(np.mean(errs), np.std(errs), np.max(np.abs(errs)))
    print('Accuracy: {:.3f}'.format(correct / total * 100.))


def test_quant_psa_resmlp_with_pkl(param_file, samples_file, compression_type=None):
    with open(param_file, 'rb') as fp:
        param_dict = pickle.load(fp)
    with open(samples_file, 'rb') as fp:
        samples_dict = pickle.load(fp)

    errs = []
    for i in range(samples_dict['input'].shape[0]):
        x = samples_dict['input'][i]
        y = resmlp(x, param_dict, 'int', is_psa=True, compression_type=compression_type)
        y_ref = samples_dict['output'][i]
        err = y - y_ref
        errs.append(err)

    errs = np.vstack(errs)
    print(np.mean(errs), np.std(errs), np.max(np.abs(errs)))


def time_quant_psa_resmlp_with_pkl(param_file, samples_file, compression_type=None):
    with open(param_file, 'rb') as fp:
        param_dict = pickle.load(fp)
    with open(samples_file, 'rb') as fp:
        samples_dict = pickle.load(fp)

    def fn():
        for i in range(samples_dict['input'].shape[0]):
            x = samples_dict['input'][i]
            y = resmlp(x, param_dict, 'int', is_psa=False, compression_type=compression_type)

    t = timeit.timeit(fn, number=10)
    print(t)


def test_all_quant_psa_resmlp_with_pkl(param_file, samples_file, compression_type=None,
        lerp_compress_points=None):

    with open(param_file, 'rb') as fp:
        param_dict = pickle.load(fp)
    with open(samples_file, 'rb') as fp:
        samples_dict = pickle.load(fp)

    if lerp_compress_points is not None:
        from psa_lut_testing import gen_lerp_based_encoded
        gen_lerp_based_encoded(param_dict, points=lerp_compress_points)

    errs = []
    correct = 0
    total = 0
    for batch_idx in tqdm(range(len(samples_dict['input']))):
        for i in range(samples_dict['input'][batch_idx].shape[0]):
            x = samples_dict['input'][batch_idx][i]
            y = resmlp(x, param_dict, 'int', is_psa=True, compression_type=compression_type)

            predicted = y.argmax(1)
            total += len(predicted)
            correct += int((predicted == samples_dict['labels'][batch_idx][i]).sum())

            y_ref = samples_dict['output'][batch_idx][i]
            err = y - y_ref
            errs.append(err)

    errs = np.vstack(errs)
    print(np.mean(errs), np.std(errs), np.max(np.abs(errs)))
    print('Accuracy: {:.3f}'.format(correct / total * 100.))


def write_params_to_h(params_pkl_file, param_h_file,
        var_type='ap_int<8>', includes=['#include <ap_int.h>'],
        weight_transpose=False):

    def layer_to_s(param, prefix):
        s = ''
        for k, v in param.items():
            var_type_tmp = var_type
            if k == 's':
                var_type_tmp = 'ap_uint<8>'
            # Skip anything to do with PSA, they are output from another function
            # in psa_lut_testing.py
            if 'psa' in prefix:
                continue
            print(prefix, k, var_type_tmp)
            if type(v) == np.array or type(v) == np.ndarray:
                s += ndarray_to_var(v, f'const {var_type_tmp}', prefix + '_' + k)
            else:
                s += var_to_var(v, f'const {var_type_tmp}', prefix + '_' + k)
        return s

    with open(params_pkl_file, 'rb') as fp:
        param_dict = pickle.load(fp)

    with open(param_h_file, 'w') as f:
        f.write(h_header(param_h_file, includes=includes))

        k = 'embed_linear'
        if weight_transpose and ('w' in param_dict[k]):
            print('Transposing Weight')
            param_dict[k]['w'] = param_dict[k]['w'].T
        f.write(layer_to_s(param_dict[k], k))

        for i in range(len(param_dict['blocks'])):
            k = 'token_mixer'
            prefix = f'blocks_{i}_{k}'
            for k2, v2 in param_dict['blocks'][i][k].items():
                if weight_transpose and ('w' in v2):
                    print('Transposing Weight')
                    v2['w'] = v2['w'].T
                f.write(layer_to_s(v2, prefix + '_' + k2))

            k = 'mlp'
            prefix = f'blocks_{i}_{k}'
            for k2, v2 in param_dict['blocks'][i][k].items():
                # if k2 == 'linear_1':
                #     v2['w'] = v2['w'].T
                if weight_transpose and ('w' in v2):
                    print('Transposing Weight')
                    v2['w'] = v2['w'].T
                f.write(layer_to_s(v2, prefix + '_' + k2))

        k = 'classifier_linear'
        if weight_transpose and ('w' in param_dict[k]):
            print('Transposing Weight')
            param_dict[k]['w'] = param_dict[k]['w'].T
        f.write(layer_to_s(param_dict[k], k))

        f.write(h_footer(param_h_file))


def write_sample_to_h(samples_pkl_file, samples_h_file, idx,
        var_type='ap_int<8>', includes=['#include <ap_int.h>'],
        data_transpose=False):

    with open(samples_pkl_file, 'rb') as fp:
        samples_dict = pickle.load(fp)

    with open(samples_h_file, 'w') as f:
        f.write(h_header(samples_h_file, includes=includes))

        k = 'input'
        v = samples_dict[k][idx]
        if data_transpose:
            v = v.T
        f.write(ndarray_to_var(v, f'const {var_type}', f'samples_{k}'))

        k = 'inter'
        for i in range(len(samples_dict[k])):
            v = samples_dict[k][i][idx]
            if data_transpose:
                v = v.T
            f.write(ndarray_to_var(v, f'const {var_type}', f'samples_{k}_{i}'))

        k = 'output'
        v = samples_dict[k][idx]
        if data_transpose:
            v = v.T
        f.write(ndarray_to_var(v, f'const {var_type}', f'samples_{k}'))

        f.write(h_footer(samples_h_file))


def write_samples_to_h(samples_pkl_file, samples_h_file, size,
        var_type='ap_int<8>', includes=['#include <ap_int.h>'],
        data_transpose=False):

    with open(samples_pkl_file, 'rb') as fp:
        samples_dict = pickle.load(fp)

    with open(samples_h_file, 'w') as f:
        f.write(h_header(samples_h_file, includes=includes))

        k = 'input'
        v = samples_dict[k][:size]
        if data_transpose:
            v = v.T
        f.write(ndarray_to_var(v, f'const {var_type}', f'samples_{k}'))

        k = 'inter'
        for i in range(len(samples_dict[k])):
            v = samples_dict[k][i][:size]
            if data_transpose:
                v = v.T
            f.write(ndarray_to_var(v, f'const {var_type}', f'samples_{k}_{i}'))

        k = 'output'
        v = samples_dict[k][:size]
        if data_transpose:
            v = v.T
        f.write(ndarray_to_var(v, f'const {var_type}', f'samples_{k}'))

        f.write(h_footer(samples_h_file))


def write_extra_samples_to_h(params_pkl_file, samples_pkl_file, samples_h_file,
        batch_idx=0,
        var_type='ap_int<8>', includes=['#include <ap_int.h>'],
        data_transpose=False):

    with open(samples_pkl_file, 'rb') as fp:
        samples_dict = pickle.load(fp)

    with open(params_pkl_file, 'rb') as fp:
        param_dict = pickle.load(fp)

    op_type = 'int'
    x = samples_dict['input'][batch_idx]
    x = linear(x, param_dict['embed_linear'], 'int')

    extra_samples = {}

    x_res = affine(x, param_dict['blocks'][0]['token_mixer']['res_affine'], op_type)
    extra_samples['token_mixer_res'] = x_res

    x = affine(x, param_dict['blocks'][0]['token_mixer']['affine_1'], op_type)
    x = x.T
    extra_samples['token_mixer_affine_1'] = x

    x = linear(x, param_dict['blocks'][0]['token_mixer']['linear'], op_type)
    extra_samples['token_mixer_linear_1'] = x

    x = x.T
    x = affine(x, param_dict['blocks'][0]['token_mixer']['affine_2'], op_type)
    extra_samples['token_mixer_affine_2'] = x

    x = add(x, x_res, op_type)
    extra_samples['token_mixer_post'] = x

    x_res = affine(x, param_dict['blocks'][0]['mlp']['res_affine'], op_type)
    extra_samples['mlp_res'] = x_res

    x = linear(x, param_dict['blocks'][0]['mlp']['linear_1'], op_type)
    extra_samples['mlp_linear_1'] = x

    x[x < 0] = 0
    extra_samples['mlp_linear_1_relu'] = x

    x = linear(x, param_dict['blocks'][0]['mlp']['linear_2'], op_type)
    extra_samples['mlp_linear_2'] = x

    x = add(x, x_res, op_type)
    extra_samples['mlp_post'] = x

    err = x - samples_dict['inter'][1][0]
    print(np.mean(err), np.std(err), np.max(np.abs(err)))

    with open(samples_h_file, 'w') as f:
        f.write(h_header(samples_h_file, includes=includes))

        for k, v in extra_samples.items():
            if data_transpose:
                v = v.T
            f.write(ndarray_to_var(v, f'const {var_type}', f'samples_{k}'))

        f.write(h_footer(samples_h_file))


def test_float_model():
    # print('\nINT LINEAR')
    # test_int_linear()

    # print('\nINT AFFINE')
    # test_int_affine()

    # test_resmlp()
    # test_resmlp_with_pkl('float_params.pkl', 'float_samples.pkl')
    pass


def test_quant_baseline():
    # parent_folder = 'hardware/baseline/resmlp_layers_4_embed_96_h_mul_2'
    parent_folder = 'hardware/baseline/resmlp_layers_4_embed_96_h_mul_1'

    # test_all_quant_resmlp_with_pkl(
    #     f'{parent_folder}/quant_params.pkl', f'{parent_folder}/all_test_quant_samples.pkl')

    # --------------------------------------------------------------------------------

    # Write a single sample (for fixed size accel)
    write_params_to_h(
        f'{parent_folder}/quant_params.pkl', f'{parent_folder}/quant_params.h',
        weight_transpose=True)
    write_sample_to_h(
        f'{parent_folder}/quant_samples.pkl', f'{parent_folder}/quant_samples.h', idx=0,
        data_transpose=True)

    # Some extra samples in the middle of computation for extra debugging
    write_extra_samples_to_h(
        f'{parent_folder}/quant_params.pkl',
        f'{parent_folder}/quant_samples.pkl',
        f'{parent_folder}/quant_samples_extra.h', batch_idx=0, data_transpose=True)

    # --------------------------------------------------------------------------------

    # # Write a batch of fixed size (for unrolled accel)
    # write_params_to_h(
    #     f'{parent_folder}/quant_params.pkl', f'{parent_folder}/quant_params.h',
    #     weight_transpose=False)
    # write_samples_to_h(
    #     f'{parent_folder}/quant_samples.pkl', f'{parent_folder}/quant_samples.h',
    #     size=8, data_transpose=False)

    # # Some extra samples in the middle of computation for extra debugging
    # write_extra_samples_to_h(
    #     f'{parent_folder}/quant_params.pkl',
    #     f'{parent_folder}/quant_samples.pkl',
    #     f'{parent_folder}/quant_samples_extra.h', batch_idx=0, data_transpose=False)

    # --------------------------------------------------------------------------------

    # parent_folder = 'hardware/resmlp_layers_4_embed_96_h_mul_1/baseline'
    # time_quant_resmlp_with_pkl(
    #     f'{parent_folder}/quant_params.pkl', f'{parent_folder}/quant_samples.pkl')
    pass


def test_quant_psa():
    # model = 'resmlp_layers_4_embed_96_h_div_2'
    model = 'resmlp_layers_4_embed_96_h_mul_1'
    # folders = [
    #     # f'hardware/lerp_psa/{model}/lerp_psa_8_256_epsilon_1e-0',
    #     # f'hardware/lerp_psa/{model}/lerp_psa_8_128_epsilon_1e-0',
    #     # f'hardware/lerp_psa/{model}/lerp_psa_8_64_epsilon_1e-0',
    #     # f'hardware/lerp_psa/{model}/lerp_psa_8_32_epsilon_1e-0',
    #     # f'hardware/lerp_psa/{model}/lerp_psa_8_16_epsilon_1e-0',
    #     # f'hardware/lerp_psa/{model}/lerp_psa_8_8_epsilon_1e-0',
    #     # f'hardware/lerp_psa/{model}/lerp_psa_8_4_epsilon_1e-0',
    #     './'
    # ]

    # for parent_folder in folders:
    #     print('')
    #     print('='*100)
    #     print(f'Checking {parent_folder}')
    #     print('='*100)
    #     print('')

    #     # for bits in reversed(range(2, 9)):
    #     for bits in reversed(range(4, 8)):
    #         print('-'*100)
    #         print(f'Grid Size = {2**bits}')

    #         if bits == 8:
    #             test_all_quant_psa_resmlp_with_pkl(
    #                 f'{parent_folder}/quant_params.pkl', f'{parent_folder}/all_test_quant_samples.pkl')
    #         else:
    #             test_all_quant_psa_resmlp_with_pkl(
    #                 f'{parent_folder}/quant_params.pkl', f'{parent_folder}/all_test_quant_samples.pkl',
    #                 compression_type='lerp', lerp_compress_points=2**bits)

    # parent_folder = f'hardware/lerp_psa/{model}/lerp_psa_8_128_epsilon_1e-0'
    parent_folder = f'hardware/lerp_psa/{model}/lerp_psa_8_64_epsilon_1e-0'

    # Write a single sample (for fixed size accel)
    write_params_to_h(
        f'{parent_folder}/quant_params.pkl', f'{parent_folder}/quant_params.h',
        weight_transpose=True)
    write_sample_to_h(
        f'{parent_folder}/quant_samples.pkl', f'{parent_folder}/quant_samples.h', idx=0,
        data_transpose=True)
    write_extra_samples_to_h(
        f'{parent_folder}/quant_params.pkl',
        f'{parent_folder}/quant_samples.pkl',
        f'{parent_folder}/quant_samples_extra.h', batch_idx=0, data_transpose=True)

    # Write a batch of fixed size (for unrolled accel)
    # write_params_to_h(
    #     f'{parent_folder}/quant_params.pkl', f'{parent_folder}/quant_params.h',
    #     weight_transpose=False)
    # write_samples_to_h(
    #     f'{parent_folder}/quant_samples.pkl', f'{parent_folder}/quant_samples.h',
    #     size=8, data_transpose=False)

    # # Some extra samples in the middle of computation for extra debugging
    # write_extra_samples_to_h(
    #     f'{parent_folder}/quant_params.pkl',
    #     f'{parent_folder}/quant_samples.pkl',
    #     f'{parent_folder}/quant_samples_extra.h', batch_idx=0)


if __name__ == "__main__":
    # test_float_model()
    test_quant_baseline()
    # test_quant_psa()
