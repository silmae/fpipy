# -*- coding: utf-8 -*-

"""Functions for manipulating raw CFA data.

Example
-------
Calculating radiances from raw data and plotting them can be done as follows::

    import xarray as xr
    import fpipy as fpi
    import matplotlib
    import os.path as osp
    from fpipy.data import house_raw


    data = house_raw() # Example raw data
    radiance = fpi.raw_to_radiance(data)
    radiance.sel(wavelength=600, method='nearest').plot()
"""

from enum import IntEnum
import numpy as np
import xarray as xr
import colour_demosaicing as cdm
from .meta import metalist


def _cfa_to_dataset(cfa, meta):
    """Combine raw CFA data with metadata into an `xr.Dataset` object.

    Parameters
    ----------
    cfa : np.ndarray
        (n,y,x) array of colour filter array images.

    meta : dict

    """

    npeaks = ('band', metalist(meta, 'npeaks'))
    wavelength = (['band', 'peak'], metalist(meta, 'wavelengths'))
    fwhm = (['band', 'peak'], metalist(meta, 'fwhms'))
    setpoints = (['band', 'setpoint'], metalist(meta, 'setpoints'))
    sinvs = (['band', 'peak', 'rgb'], metalist(meta, 'sinvs'))

    return xr.Dataset(
        coords={'peak': [1, 2, 3],
                'setpoint': [1, 2, 3],
                'rgb': ['R', 'G', 'B']},

        data_vars={'cfa': cfa,
                   'npeaks': npeaks,
                   'wavelength': wavelength,
                   'fwhm': fwhm,
                   'setpoints': setpoints,
                   'sinvs': sinvs},

        attrs={'fpi_temperature': meta.getfloat('Header', 'fpi temperature'),
               'description': meta.get('Header', 'description').strip('"'),
               'dark_layer_included':
                   meta.getboolean('Header', 'dark layer included'),
               'gain': meta.getfloat('Image0', 'gain'),
               'exposure': meta.getfloat('Image0', 'exposure time (ms)'),
               'bayer_pattern': meta.getint('Image0', 'bayer pattern')})


def cfa_stack_to_da(
        cfa,
        pattern,
        index=None,
        x=None,
        y=None,
        ):
    """Check metadata validity and form a DataArray from a stack of FPI images.

    Parameters
    ----------
    cfa: np.ndarray
        (n_indices, height, width) stack of CFA images taken through a
        Fabry-Perot tunable filter.

    pattern: str or BayerPattern
        Bayer filter pattern of the camera CFA.

    index: array-like, optional
        1-d array of unique indices identifying settings used for each image.
        Defaults to a vector of integers from 0 to n_indices.

    x: array-like, optional
        1-D array of unique x coordinates.
        Defaults to a vector of pixel centers from 0.5 to width - 0.5.

    y: array-like, optional
        1-D array of unique y coordinates.
        Defaults to a vector of pixel centers from 0.5 to height - 0.5.


    Returns
    -------
    xr.DataArray
        `xr.DataArray` containing the CFA stack with labeled dimensions.
    """

    if index is None:
        index = np.arange(0, cfa.shape[0])

    if x is None:
        x = np.arange(0, cfa.shape[2]) + 0.5

    if y is None:
        y = np.arange(0, cfa.shape[1]) + 0.5

    cfa_da = xr.DataArray(
            cfa,
            dims=('index', 'y', 'x'),
            coords={'index': index, 'y': y, 'x': x},
            attrs={'pattern': str(pattern)}
            )
    return cfa_da


def raw_to_radiance(dataset, pattern=None, dm_method='bilinear'):
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
        layers = dataset.cfa
        raise UserWarning('Dark layer is not included in dataset!')

    if pattern is None:
        pattern = dataset.bayer_pattern

    radiance = {}

    for layer in layers:
        demo = demosaic(layer, pattern, dm_method)

        for n in range(1, dataset.npeaks.sel(band=layer.band).values + 1):
            data = dataset.sel(band=layer.band, peak=n)

            rad = data.sinvs.dot(demo)/data.exposure

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
    radiance.name = 'radiance'

    return radiance


def subtract_dark(array, dark=None):
    """Substracts dark reference from other image layers.

    Parameters
    ----------
    array : xarray.DataArray

    dark : xarray.DataArray, optional
        This is typically included as the first layer of array.

    Returns
    -------
    refarray : xarray.DataArray
        Layers from which the dark reference layer has been substracted.
        Resulting array will have dtype float64.
    """

    if dark is None:
        output = array[1:] - array[0]
    else:
        output = array[:] - dark

    output.values[output.values < 0] = 0
    return output


class BayerPattern(IntEnum):
    """Enumeration of the Bayer Patterns as used by FPI headers."""
    GBRG = 0
    GRBG = 1
    BGGR = 2
    RGGB = 3

    def __str__(self):
        return self.name

    @classmethod
    def get(self, pattern):
        try:
            return self[pattern.upper()]
        except (KeyError, AttributeError):
            return self(pattern)


def demosaic(cfa, pattern, dm_method):
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

    dm_methods = {
        'bilinear': cdm.demosaicing_CFA_Bayer_bilinear,
        'Malvar2004': cdm.demosaicing_CFA_Bayer_Malvar2004,
        'Menon2007': cdm.demosaicing_CFA_Bayer_Menon2007,
        }
    dm_alg = dm_methods[dm_method]

    return xr.DataArray(
        dm_alg(cfa, pattern_name),
        dims=['y', 'x', 'rgb'],
        coords={'y': cfa.y, 'x': cfa.x, 'rgb': ['R', 'G', 'B']},
        attrs=cfa.attrs)
