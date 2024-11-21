import numpy as np
# import torch


def ndarray_to_str(x, brkt_op='{', brkt_cl='}', delim=','):
    ndims = len(x.shape)
    nelem = x.shape[0]
    s = brkt_op
    for i in range(nelem):
        if ndims <= 1:
            s += x[i].astype(str)
            if i < nelem - 1:
                s += delim + ' '
        else:
            s += ndarray_to_str(x[i])
            if i < nelem - 1:
                s += delim + '\n'
    s += brkt_cl
    return s


def ndarray_to_var(x, vartype, varname):
    body_str = ndarray_to_str(x)

    s = "\n{} {}".format(vartype, varname)
    for z in np.shape(x):
        s = s + "[{}]".format(z)
    s = s + " = \n" + body_str + ";\n"
    return s


def var_to_var(x, vartype, varname):
    body_str = str(x)

    s = "\n{} {}".format(vartype, varname)
    s = s + " = " + body_str + ";\n"
    return s


def h_header(filename, includes=['#include <stdint.h>']):
    define_tag = filename.upper().replace('.', '_').replace('/', '_').replace('-', '_')
    s = '#ifndef {}\n'.format(define_tag)
    s += '#define {}\n'.format(define_tag)
    s += '\n' + '\n'.join(includes) + '\n'
    return s


def h_footer(filename):
    define_tag = filename.upper().replace('.', '_').replace('/', '_')
    s = '\n#endif\n'
    return s


def main():
    FILENAME = 'params.h'
    np.random.seed(2022)

    # Use full-precision print option
    np.set_printoptions(floatmode='maxprec')

    # x = torch.randn(2,4,3,3).numpy()
    # x = np.random.randn(2, 4, 3, 3).astype(np.float32)

    w = (np.random.randn(128) * 128/3).clip(-128, 127).astype(np.int8)
    pe = (np.random.randn(128) * 128/(3*4)).clip(-128, 127).astype(np.int8)

    print(w)

    header = h_header(FILENAME)
    s1 = ndarray_to_var(w, 'const int8_t', 'layer_0_w')
    s2 = ndarray_to_var(pe, 'const int8_t', 'layer_0_pe')
    # v = var_to_var(np.float32(2**(1./2)), 'const float', 'alpha')
    footer = h_footer(FILENAME)

    with open(FILENAME, 'w') as f:
        f.write(header)
        f.write(s1)
        f.write(s2)
        # f.write(v)
        f.write(footer)


if __name__ == '__main__':
    main()
