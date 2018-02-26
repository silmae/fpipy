# -*- coding: utf-8 -*-

""" Error estimates and calculations. """

from .raw import subtract_dark, BayerPattern
import xarray as xr
from scipy.signal import fftconvolve

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
             [1,  0, 1]]) / 8

    G = fftconvolve(CFA * G_m, K_err, mode='same')

    return G

def raw_to_radiance(dataset, pattern=None):
    """Performs demosaicing and computes radiance from RGB values.

    Parameters
    ----------
    dataset : xarray.Dataset
        Requires data to be found via dataset.cfa, dataset.npeaks,
        dataset.sinvs, dataset.wavelength, dataset.fwhm and dataset.exposure.

    pattern : BayerPattern or str, optional
        Bayer pattern used to demosaic the CFA.
        Can be supplied to override the file metadata value in cases where it
        is missing or incorrect.

    dm_method : str, optional
        **{'bilinear', 'DDFAPD', 'Malvar2004', 'Menon2007'}**
        Demosaicing method. Default is bilinear. See the `colour_demosaicing`
        package for more info on the different methods.

    Returns
    -------
    radiance : xarray.DataArray
        Includes computed radiance sorted by wavelength
        and with x, y, wavelength and fwhm as coordinates.
        Passes along relevant attributes from input dataset.
    """

    if dataset.dark_layer_included:
        layers = subtract_dark(dataset.cfa.astype('int16'))
    else:
        raise UserWarning('Dark layer is not included in dataset!')
        layers = dataset.cfa

    if pattern is None:
        pattern = dataset.bayer_pattern

    radiance = {}

    for layer in layers:
        demo = demosaic(layer, pattern)

        for n in range(1, dataset.npeaks.sel(band=layer.band).values + 1):
            data = dataset.sel(band=layer.band, peak=n)

            rad = (data.sinvs.sel(rgb='G') * demo)/data.exposure

            rad.coords['wavelength'] = data.wavelength
            rad.coords['fwhm'] = data.fwhm
            rad = rad.drop('peak')
            rad = rad.drop('band')
            radiance[float(rad.wavelength)] = rad

    radiance = xr.concat([radiance[key] for key in sorted(radiance)],
                         dim='wavelength',
                         coords=['wavelength', 'fwhm'])
    radiance.attrs = {key: value for key, value in dataset.attrs.items()
                      if key not in ['dark_layer_included', 'bayer_pattern']}

    return radiance


def demosaic(cfa, pattern):
    """Perform demosaicing on a DataArray.

    Parameters
    ----------
    cfa : xarray.DataArray

    pattern : BayerPattern or str, optional
        Bayer pattern used to demosaic the CFA.
        Can be supplied to override the file metadata value in cases where it
        is missing or incorrect.

    dm_method : str

    Returns
    -------
    xarray.DataArray
    """
    pattern_name = BayerPattern.get(pattern).name

    return xr.DataArray(
        interp_err_lv_bilin(cfa, pattern_name),
        dims=['y', 'x'],
        coords={'y': cfa.y, 'x': cfa.x},
        attrs=cfa.attrs)
