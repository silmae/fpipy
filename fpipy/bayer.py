from enum import IntEnum
import numpy as np
import xarray as xr
from scipy.ndimage import convolve
from . import conventions as c


def find_bayer_pattern(ds):
    """Retrieve Bayer pattern string from image data."""
    attrs = ds[c.dark_corrected_cfa_data].attrs
    if c.cfa_pattern_data in ds:
        pattern = ds[c.cfa_pattern_data]
    elif c.genicam_pattern_data in ds:
        pattern = ds[c.genicam_pattern_data]
    elif c.cfa_pattern_data in attrs:
        pattern = attrs[c.cfa_pattern_data]
    elif c.genicam_pattern_data in attrs:
        pattern = attrs[c.genicam_pattern_data]
    else:
        raise ValueError('Bayer pattern not specified.')
    return pattern


def bayer_masks(raw):
    """Create Bayer filter mosaic colour separation masks.

    Parameters
    ----------
    raw : xr.Dataset
        Dataset containing CFA and associated pattern data.

    Returns
    -------
    xr.DataArray
        (color, y, x) array of Bayer masks
    """
    shape = (raw[c.height_coord].size, raw[c.width_coord].size)
    pattern = np.unique(find_bayer_pattern(raw)).item()

    masks, coords = _bayer_masks(shape, pattern)
    res = xr.DataArray(
        masks,
        dims=(c.colour_coord, c.height_coord, c.width_coord),
        coords={
            c.colour_coord: coords,
            c.height_coord: raw[c.height_coord],
            c.width_coord: raw[c.width_coord],
            }
    )
    return res


def _bayer_masks(shape, pattern):
    pattern = BayerPattern.get(pattern).name

    channels = dict((channel, np.zeros(shape, dtype=np.bool))
                    for channel in 'RGB')
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1

    masks = np.stack(list(channels.values()), axis=0)
    return masks, list(channels.keys())


def inversion_method(pixelformat):
    """Select an efficient radiance inversion method based on bit format.
    """
    methods = {
        'BayerGR12': demosaic_and_invert_12bit_low,
        'BayerRG12': demosaic_and_invert_12bit_low,
        'BayerGB12': demosaic_and_invert_12bit_low,
        'BayerBG12': demosaic_and_invert_12bit_low,
        # We assume 16 bit formats are actually from a 12bit ADC, stored
        # in the high bits (since that is the case for FLIR/PtGrey cameras)
        'BayerGR16': demosaic_and_invert_12bit_high,
        'BayerRG16': demosaic_and_invert_12bit_high,
        'BayerGB16': demosaic_and_invert_12bit_high,
        'BayerBG16': demosaic_and_invert_12bit_high,
        }
    default = demosaic_and_invert_float

    return methods.get(pixelformat, default)


def demosaic_and_invert_float(mosaic, masks, sinvs, exposure):
    """Compute radiances from a Bayer filter mosaic.

    Demosaics the mosaic image and computes the radiance(s).
    See `demosaic_bilin` and `invert_RGB` for more information.
    """
    rad = invert_RGB(
        demosaic_bilin_float(mosaic, masks),
        sinvs,
        exposure
    )
    return rad


def demosaic_and_invert_12bit_high(mosaic, masks, sinvs, exposure):
    """Compute radiances from a Bayer filter mosaic.

    Demosaics the mosaic image and computes the radiance(s).
    See `demosaic_bilin` and `invert_RGB` for more information.
    """
    mosaic = np.right_shift(mosaic, 2)
    rad = invert_RGB(
        demosaic_bilin_12bit(mosaic, masks),
        sinvs,
        exposure
    )
    return rad * 4


def demosaic_and_invert_12bit_low(mosaic, masks, sinvs, exposure):
    """Compute radiances from a Bayer filter mosaic.

    Demosaics the mosaic image and computes the radiance(s).
    See `demosaic_bilin` and `invert_RGB` for more information.
    """
    mosaic = np.left_shift(mosaic, 2)
    rad = invert_RGB(
        demosaic_bilin_12bit(mosaic, masks),
        sinvs,
        exposure
    )
    return rad / 4


def invert_RGB(rgbs, sinvs, exposure):
    """Compute radiance(s) from RGB data

    Parameters
    ----------
    rgbs : np.ndarray
        (3, y, x) array of RGB images.

    sinvs : np.ndarray
        (peak, 3) float array of inversion coefficients for computing
        the pseudoradiance from the RGB values.

    exposure : float
        Exposure of the RGB data, div

    Returns
    -------
    np.ndarray
        (peak, y, x) float array of pseudoradiance values.
    """
    return np.tensordot(sinvs, rgbs, axes=1) / exposure


def demosaic_bilin_float(cfa, masks):
    """Demosaics and computes (pseudo)radiance using given arrays.

    Parameters
    ----------
    cfa : np.ndarray
        (y, x) array of CFA data.
    masks : np.ndarray
        (3, y, x) boolean array of R, G, and B mask arrays.

    Returns
    -------
    rgb : np.ndarray
        (3, y, x) demosaiced RGB image
    """
    rgbs = cfa * masks
    g_krnl = np.array(
        [[0, 1, 0],
         [1, 4, 1],
         [0, 1, 0]], dtype=np.float64) / 4

    rb_krnl = np.array(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]], dtype=np.float64) / 4

    R = convolve(rgbs[0, ::], rb_krnl, mode='mirror', output=np.float64)
    G = convolve(rgbs[1, ::], g_krnl, mode='mirror', output=np.float64)
    B = convolve(rgbs[2, ::], rb_krnl, mode='mirror', output=np.float64)
    return np.stack([R, G, B], axis=0)


def demosaic_bilin_12bit_scipy(cfa, masks):
    """Demosaics and computes (pseudo)radiance using given arrays.

    Parameters
    ----------
    cfa : np.ndarray
        (y, x) uint16 array of natively 12-bit CFA data assumed to be
        packed as values in the middle 12 bits of uint16 (4 to 16380).
    masks : np.ndarray
        (3, y, x) boolean array of R, G, and B mask arrays.

    Returns
    -------
    rgb : np.ndarray
        (3, y, x) demosaiced RGB image
    """
    rgbs = cfa * masks
    g_krnl = np.array(
        [[0, 1, 0],
         [1, 4, 1],
         [0, 1, 0]], dtype=np.uint16)

    rb_krnl = np.array(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]], dtype=np.uint16)

    R = convolve(rgbs[0, ::], rb_krnl, mode='mirror', output=np.uint16) >> 2
    G = convolve(rgbs[1, ::], g_krnl, mode='mirror', output=np.uint16) >> 2
    B = convolve(rgbs[2, ::], rb_krnl, mode='mirror', output=np.uint16) >> 2

    return np.stack([R, G, B], axis=0)


def demosaic_bilin_12bit(cfa, masks):
    """Demosaics and computes (pseudo)radiance using given arrays.

    Parameters
    ----------
    cfa : np.ndarray
        (y, x) uint16 array of natively 12-bit CFA data assumed to be
        packed as values in the middle 12 bits of uint16 (4 to 16380).
    masks : np.ndarray
        (3, y, x) boolean array of R, G, and B mask arrays.

    Returns
    -------
    rgb : np.ndarray
        (3, y, x) demosaiced RGB image
    """
    res = cfa * masks
    rgbs = np.pad(res, [(0, 0), (1, 1), (1, 1)], mode='reflect')

    res = res << 2

    # Red channel: convolution with kernel
    # [[1, 2, 1]]
    #  [2, 4, 2]
    #  [1, 2, 1]]
    res[0, ::] += rgbs[0,  :-2, 1:-1] << 1  # noqa: E203
    res[0, ::] += rgbs[0,   2:, 1:-1] << 1  # noqa: E203
    res[0, ::] += rgbs[0, 1:-1,  :-2] << 1  # noqa: E203
    res[0, ::] += rgbs[0, 1:-1,   2:] << 1  # noqa: E203
    res[0, ::] += rgbs[0,  :-2,  :-2]  # noqa: E203
    res[0, ::] += rgbs[0,   2:,  :-2]  # noqa: E203
    res[0, ::] += rgbs[0,  :-2,   2:]  # noqa: E203
    res[0, ::] += rgbs[0,   2:,   2:]  # noqa: E203

    # Green channel: convolution with kernel
    # [[0, 1, 0]]
    #  [1, 4, 1]
    #  [0, 1, 0]]
    res[1, ::] += rgbs[1,  :-2, 1:-1]  # noqa: E203
    res[1, ::] += rgbs[1,   2:, 1:-1]  # noqa: E203
    res[1, ::] += rgbs[1, 1:-1,  :-2]  # noqa: E203
    res[1, ::] += rgbs[1, 1:-1,   2:]  # noqa: E203

    # Blue channel: convolution with kernel
    # [[1, 2, 1]]
    #  [2, 4, 2]
    #  [1, 2, 1]]
    res[2, ::] += rgbs[2,  :-2, 1:-1] << 1  # noqa: E203
    res[2, ::] += rgbs[2,   2:, 1:-1] << 1  # noqa: E203
    res[2, ::] += rgbs[2, 1:-1,  :-2] << 1  # noqa: E203
    res[2, ::] += rgbs[2, 1:-1,   2:] << 1  # noqa: E203
    res[2, ::] += rgbs[2,  :-2,  :-2]  # noqa: E203
    res[2, ::] += rgbs[2,   2:,  :-2]  # noqa: E203
    res[2, ::] += rgbs[2,  :-2,   2:]  # noqa: E203
    res[2, ::] += rgbs[2,   2:,   2:]  # noqa: E203

    res = res >> 2
    return res


class BayerPattern(IntEnum):
    """Enumeration of the Bayer Patterns as used by FPI headers."""
    GBRG = 0
    GRBG = 1
    BGGR = 2
    RGGB = 3

    # Lowercase aliases.
    gbrg = 0
    grbg = 1
    bggr = 2
    rggb = 3

    # Aliases (GenICam PixelColorFilter values)
    BayerGB = 0
    BayerGR = 1
    BayerBG = 2
    BayerRG = 3

    @classmethod
    def get(self, pattern):
        try:
            return self[pattern]
        except (KeyError, AttributeError):
            return self(pattern)

    def __str__(self):
        return self.name
