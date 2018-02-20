# -*- coding: utf-8 -*-
"""
Bilinear Bayer CFA Demosaicing
==============================
*Bayer* CFA (Colour Filter Array) bilinear demosaicing.
This code has been roughly derived from colour_demosaicing for the purpose of
matching the methods that our camera's own demosaicing code uses.

References
----------
-   :cite:`Losson2010c` : Losson, O., Macaire, L., & Yang, Y. (2010).
    Comparison of Color Demosaicing Methods. In Advances in Imaging and
    Electron Physics (Vol. 162, pp. 173-265). doi:10.1016/S1076-5670(10)62005-8
"""

from __future__ import division, unicode_literals

import numpy as np

#from scipy.ndimage.filters import convolve
from scipy.signal import fftconvolve

from colour.utilities import tstack

#__author__ = 'Colour Developers'
#__copyright__ = 'Copyright (C) 2015-2018 - Colour Developers'
#__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
#__maintainer__ = 'Colour Developers'
#__email__ = 'colour-science@googlegroups.com'
#__status__ = 'Production'

__all__ = ['demosaicing_CFA_Bayer_uglybilinear']


def masks_CFA_Bayer_twogreens(shape, pattern='RGGB'):
    """
    Returns the *Bayer* CFA red, g_1, g_2 and blue masks for given pattern.
    
    Parameters
    ----------
    shape : array_like
        Dimensions of the *Bayer* CFA.
    pattern : unicode, optional
        **{'RGGB', 'BGGR', 'GRBG', 'GBRG'}**,
        Arrangement of the colour filters on the pixel array.
        
    Returns
    -------
    tuple
        *Bayer* CFA red, g_1, g_2 and blue masks.
    """

    pattern = 'RXYB'

    channels = dict((channel, np.zeros(shape)) for channel in 'RXYB')
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1

    return tuple(channels[c].astype(bool) for c in 'RXYB')


def demosaicing_CFA_Bayer_uglybilinear(CFA, pattern='RGGB'):
    """
    Returns the demosaiced *RGB* colourspace array from given *Bayer* CFA using
    bilinear interpolation that averages over two separate green channels.

    Parameters
    ----------
    CFA : array_like
        *Bayer* CFA.
    pattern : unicode, optional
        **{'RGGB', 'BGGR', 'GRBG', 'GBRG'}**,
        Arrangement of the colour filters on the pixel array.

    Returns
    -------
    ndarray
        *RGB* colourspace array.

    Notes
    -----
    -   The definition output is not clipped in range [0, 1] : this allows for
        direct HDRI / radiance image generation on *Bayer* CFA data and post
        demosaicing of the high dynamic range data as showcased in this
        `Jupyter Notebook <https://github.com/colour-science/colour-hdri/\
blob/develop/colour_hdri/examples/\
examples_merge_from_raw_files_with_post_demosaicing.ipynb>`_.

    References
    ----------
    -   :cite:`Losson2010c`
    """

    CFA = np.asarray(CFA).astype('float32')
    R_m, G_1m, G_2m, B_m = masks_CFA_Bayer_twogreens(CFA.shape, pattern)

    H = np.asarray(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]], dtype='float32') / 4  # yapf: disable

    R = fftconvolve(CFA * R_m, H, mode='same')
    G_1 = fftconvolve(CFA * G_1m, H, mode='same')
    G_2 = fftconvolve(CFA * G_2m, H, mode='same')
    B = fftconvolve(CFA * B_m, H, mode='same')

    G = (np.rint(G_1) + np.rint(G_2))/2.

    return np.rint(tstack((R, G, B))).astype('uint16')
