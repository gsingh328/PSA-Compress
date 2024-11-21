import numpy as np
import math
import matplotlib.pyplot as plt


def lerp_compress(lut, points, base_bits=8, report_err=False, append=True):
    assert points < lut.shape[-1]

    # Get our sample points (always include the last element)
    step_size = int(math.ceil(lut.shape[-1] / points))
    lerp_lut = lut[...,0::step_size]
    if append:
        lerp_lut = np.append(lerp_lut, lut[...,-2:-1], axis=-1)
        assert lerp_lut.shape[-1] == points + 1
    else:
        assert lerp_lut.shape[-1] == points

    # Get the approximation error
    if report_err:
        x = np.linspace(0, 2**base_bits-1, 2**base_bits).astype(int)

        mbs_bits = int(math.ceil(math.log2(points)))
        lsb_bits = base_bits - mbs_bits

        x_msb = x >> lsb_bits
        x_lsb = x & (2**lsb_bits - 1)
        if not append:
            x_msb = np.clip(x_msb, 0, 2**mbs_bits - 2)

        y0 = lerp_lut[...,x_msb]
        y1 = lerp_lut[...,x_msb+1]
        y = (((y1 - y0) * x_lsb) >> (lsb_bits - 1))
        y_rnd_factor = y & 1
        y = (y >> 1) + y_rnd_factor
        y += y0

        err = lut - y
        print('Error Mean, STD, Min, Max\n{:.3f}, {:.3f}, {}, {}'.format(
            np.mean(err), np.std(err), np.min(err), np.max(err)))

        # for i in range(lut.shape[0]):
        #     plt.clf()
        #     plt.plot(x, y[i,:], color='red')
        #     plt.plot(x, lut[i,:], color='blue')
        #     plt.savefig(f'graphs/{i}.png')
        # exit(1)
    return lerp_lut
