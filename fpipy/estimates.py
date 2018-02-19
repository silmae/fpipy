# -*- coding: utf-8 -*-

""" Error estimates and calculations. """


def interp_err_lv_bilin(CFA, pattern):
    """Interpolation error resulting from the Labview implementation.

    The existing Labview implementation interpolates two green layers
    separately and takes their average afterwards instead of interpolating a
    combined green layer. The difference of the methods is

    (G1 * K_b1 + G2 * K_b1)/2 - (G1 + G2) * K_b2
        = (G1 + G2) * (K_b1 / 2 - K_b2),

    with (*) marking convolution, G1, G2 the green layers and K_b1 and K_b2 the
    kernels corresponding to bilinear interpolation for 1 in 4 and 1 in 2 pixel
    cases, respectively. Since the actual kernels used are

    K_b1 = [[1 2 1]
            [2 4 1]
            [1 2 1]] / 4

    and

    K_b2 = [[0 1 0]
            [1 4 1]
            [0 1 0]] / 4,

    which are not a factor of two apart, this results in error whenever there
    is a gradient in the green layer. This function computes the error for a
    given CFA image and Bayer pattern.

    Code modified from bilinear.py in colour_demosaicing.
    """

    import numpy as np
    from colour_demosaicing.bayer.masks import masks_CFA_Bayer
    from scipy.ndimage.filters import convolve

    _, G_m, _ = masks_CFA_Bayer(CFA.shape, pattern)

    CFA = np.asarray(CFA)
    K_err = np.asarray(
            [[1,  0, 1],
             [0, -4, 0],
             [1,  0, 1]]) / 2

    G = convolve(CFA * G_m, K_err)

    return G
