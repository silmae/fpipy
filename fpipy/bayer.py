from enum import IntEnum
import numpy as np
import xarray as xr
import colour_demosaicing as cdm
from scipy.ndimage import convolve
from . import conventions as c
from .utils import _drop_variable


def _raw_to_rgb(raw, keep_variables=None):
    """Demosaic a dataset of CFA data.

    Parameters
    ----------
    raw: xr.Dataset
        Dataset containing `c.dark_corrected_cfa_data` and mosaic pattern
        information either as a variable or an attribute of the cfa variable.

    keep_variables: list-like, optional
        List of variables to keep in the result, default None.
        If you wish to keep the raw CFA data, pass a list including
        `fpipy.conventions.cfa_data`.

    Returns
    -------
    res: xr.Dataset
        Dataset containing the demosaiced R, G and B layers as a variable.
    """
    raw = raw.copy()
    attrs = raw[c.dark_corrected_cfa_data].attrs
    if c.cfa_pattern_data in raw:
        pattern = str(raw[c.cfa_pattern_data].values)
    elif c.genicam_pattern_data in raw:
        pattern = str(raw[c.genicam_pattern_data].values)
    elif c.cfa_pattern_data in attrs:
        pattern = str(attrs[c.cfa_pattern_data])
    elif c.genicam_pattern_data in attrs:
        pattern = str(attrs[c.genicam_pattern_data])
    else:
        raise ValueError('Bayer pattern not specified.')

    raw[c.rgb_data] = demosaic_cdm(
            raw[c.dark_corrected_cfa_data],
            pattern,
            )

    return _drop_variable(raw, c.dark_corrected_cfa_data, keep_variables)


def demosaic_bilin(raw, keep_variables=None):
    raw = raw.copy()
    raw['bilin_kernel'] = bilin_kernel()
    raw['bayer_mask'] = create_mask(raw)

    raw[c.rgb_data] = raw.groupby(c.colour_coord).apply(_convolve)

    raw = _drop_variable(raw, c.dark_corrected_cfa_data, keep_variables)
    raw = _drop_variable(raw, 'bilin_kernel', keep_variables)
    raw = _drop_variable(raw, 'bayer_mask', keep_variables)
    return raw


def _convolve(raw):
    res = xr.apply_ufunc(
        convolve,
        raw[c.dark_corrected_cfa_data] * raw['bayer_mask'],
        raw['bilin_kernel'],
        kwargs={'mode': 'mirror'},
        dask='parallelized',
        input_core_dims=[[], ['krnl_x', 'krnl_y']],
        output_dtypes=[np.uint16],
    ) // 4
    return res


def create_mask(raw):
    attrs = raw[c.dark_corrected_cfa_data].attrs
    if c.cfa_pattern_data in raw:
        pattern = str(raw[c.cfa_pattern_data].values)
    elif c.genicam_pattern_data in raw:
        pattern = str(raw[c.genicam_pattern_data].values)
    elif c.cfa_pattern_data in attrs:
        pattern = str(attrs[c.cfa_pattern_data])
    elif c.genicam_pattern_data in attrs:
        pattern = str(attrs[c.genicam_pattern_data])
    else:
        raise ValueError('Bayer pattern not specified.')
    pattern = BayerPattern.get(pattern).name
    shape = raw[c.dark_corrected_cfa_data].shape

    res = xr.DataArray(
            np.stack(cdm.bayer.masks_CFA_Bayer(shape, pattern), axis=-1),
            dims=c.RGB_dims,
            coords={c.colour_coord: ['R', 'G', 'B']}
    )
    return res


def bilin_kernel():
    H_G = np.array(
        [[0, 1, 0],
         [1, 4, 1],
         [0, 1, 0]], dtype=np.uint16)

    H_RB = np.array(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]], dtype=np.uint16)

    krnl = xr.DataArray(
        np.stack([H_RB, H_G, H_RB], axis=-1),
        dims=('krnl_x', 'krnl_y', c.colour_coord)
    )
    return krnl


def demosaic_cdm(cfa, pattern):
    """Perform demosaicing on a DataArray.

    Parameters
    ----------
    cfa: xarray.DataArray
        Array containing a stack of CFA images.

    pattern: BayerPattern or str
        Bayer pattern used to demosaic the CFA.

    dm_method: str

    Returns
    -------
    xarray.DataArray
    """

    res = xr.apply_ufunc(
        cdm.demosaicing_CFA_Bayer_bilinear,
        cfa,
        kwargs=dict(pattern=pattern),
        input_core_dims=[(c.height_coord, c.width_coord)],
        output_core_dims=[(c.RGB_dims)],
        dask='parallelized',
        output_dtypes=[np.float64],
        output_sizes={c.colour_coord: 3}
        )
    res.coords[c.colour_coord] = ['R', 'G', 'B']
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
