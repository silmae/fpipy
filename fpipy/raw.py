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


    data = house_raw() # Example raw data (including dark current)
    data = subtract_dark(data)
    radiance = fpi.raw_to_radiance(data)
    radiance.sel(wavelength=600, method='nearest').plot()
"""

from enum import IntEnum
import numpy as np
import xarray as xr
import colour_demosaicing as cdm


def _cfa_to_dataset(
        cfa,
        dark,
        npeaks,
        wavelength,
        fwhm,
        setpoints,
        sinvs,
        attrs=None
        ):
    """Combine raw CFA data with metadata into an `xr.Dataset` object.

    Parameters
    ----------
    cfa: xr.DataArray
        (n,y,x) array of colour filter array images with labeled dimensions
        index, x and y.
        See `cfa_stack_to_da`.

    dark: xr.DataArray

    npeaks: xr.DataArray
        Number of separate passbands (peaks) for each index.

    wavelength: xr.DataArray
        Wavelengths corresponding to each peak for each index.

    fwhm: xr.DataArray
        FWHMs corresponding to each peak for each index.

    setpoints: xr.DataArray
        1st setpoints (voltage values sent to the PFPI driver)
        for each index.

    sinvs: xr.DataArray
        Inversion coefficients for radiance calculation corresponding
        to each index, peak and colour.

    attrs: dict, optional
        Extra attributes to add to the dataset.

    Returns
    -------
    xr.Dataset

    """

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
        attrs=attrs
        )


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
            attrs={'bayer_pattern': str(pattern)}
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

    layers = dataset.cfa

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


def subtract_dark(
        data,
        dark,
        data_var=None,
        dc_attr='includes_dark_current'):
    """Subtracts dark current reference from other image layers.

    Subtracts a dark reference frame from all the layers in the given data
    and clamps any negative values in the result to zero.

    Parameters
    ----------
    data : xarray.DataArray or xarray.DataSet
        Dataset containing the raw images (including dark current)
        either directly (if DataArray) or as the given data_var.

    dark : array-like
        Dark current reference measurement.

    data_var : str, optional
        If given, attempts to operate only on data[data_var] instead of
        data.

    dc_attr : str, optional
        Attribute to use for checking whether the data includes dark current,
        and to set to False afterwards.
        Default: 'includes_dark_current'

    Returns
    -------
    refarray : xarray.DataArray or xarray.Dataset
        Data from which the dark current reference has been subtracted.
        Resulting array will have dtype float64, with negative values
        clamped to 0.
        If an included dark reference was supplied, it is removed from the
        result.

    """

    if data_var is None:
        try:
            if not data[dc_attr]:
                raise UserWarning(
                    'Data already has {} set to false!'.format(dc_attr))
        except:
            pass

        data = _subtract_dark(data, dark)
        data.attrs[dc_attr] = False

    else:
        try:
            if not data[data_var].attrs[dc_attr]:
                raise UserWarning(
                    'Data already has {} set to false!'.format(dc_attr))
        except:
            pass

        data[data_var] = _subtract_dark(data[data_var], dark)
        data[data_var].attrs[dc_attr] = False

    return data


def _subtract_dark(array, dark):
    """Subtract dark from array and clip to non-negative values."""
    result = array.astype(np.float64) - dark.astype(np.float64)
    result.clip(min=0)
    return result


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
