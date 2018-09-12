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

from warnings import warn
from enum import IntEnum
import numpy as np
import xarray as xr
import colour_demosaicing as cdm
from . import conventions as c


def _cfa_to_dataset(
        *,
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
        coords={c.peak_coord: [1, 2, 3],
                c.setpoint_coord: [1, 2, 3],
                c.colour_coord: ['R', 'G', 'B']},

        data_vars={c.cfa_data: cfa,
                   c.npeaks: npeaks,
                   c.wavelength_data: wavelength,
                   c.fwhm_data: fwhm,
                   c.setpoint_data: setpoints,
                   c.sinv_data: sinvs},
        attrs=attrs
        )


def cfa_stack_to_da(
        *,
        cfa,
        pattern,
        includes_dark_current,
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

    includes_dark_current: bool
        Whether or not the data values include dark current.

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
    attrs = {
            c.cfa_pattern_attribute: str(pattern),
            c.dc_included_attr: int(includes_dark_current),
            }
    cfa_da = xr.DataArray(
            cfa,
            dims=c.cfa_dims,
            coords={
                c.image_index: index,
                c.height_coord: y,
                c.width_coord: x,
                },
            attrs=attrs,
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


def raw_to_radiance2(dataset, dm_method='bilinear'):

    # Subtract dark reference from the data
    dataset[c.cfa_data] = subtract_dark(
            dataset[c.cfa_data],
            dataset[c.dark_reference_data]
            )

    # Workaround until xarray supports passing of positional parameters
    def raw_to_rgb(raw):
        return _raw_to_rgb(raw, dm_method)

    # Demosaic each image
    dataset[c.rgb_data] = dataset.groupby(
            c.image_index
            ).apply(raw_to_rgb)

    # Create a new band coordinate and use it to select only existing peaks
    dataset = dataset.stack(**{c.band_index: (c.image_index, c.peak_coord)})
    dataset = dataset.sel(
        **{c.band_index: dataset[c.peak_coord] <= dataset[c.number_of_peaks]}
        )

    # Preserve other data by setting them as coordinates
    preserved_data = [
            c.wavelength_data,
            c.fwhm_data,
            c.camera_exposure,
            c.camera_gain,
            ]
    dataset = dataset.set_coords([c for c in preserved_data if c in dataset])

    # Calculate radiances from each RGB image
    radiances = dataset.groupby(c.band_index).apply(_rgb_to_rad)
    radiances = radiances.reset_coords(preserved_data)

    # Sort by wavelengths and reassign band coordinate
    radiances = radiances.sortby(c.wavelength_data)
    radiances = radiances.assign_coords(
            **{c.band_index: dataset[c.band_index]}
            )
    return radiances

import gc
def _raw_to_rgb(raw, dm_method):
    """Demosaic a dataset of CFA data.

    Parameters
    ----------
    raw: xr.Dataset
        Dataset containing a single CFA image and Bayer pattern information.

    Returns
    -------
    xr.DataArray
        DataArray containing the demosaiced R, G and B layers.
    """
    if c.cfa_pattern_attribute in raw[c.cfa_data].attrs:
        pattern = str(raw[c.cfa_data].attrs[c.cfa_pattern_attribute])
    else:
        pattern = str(raw[c.cfa_pattern_attribute].values)

    rgb = demosaic(
            raw[c.cfa_data],
            pattern,
            dm_method
            )
    return rgb


def _rgb_to_rad(rgb):
    """Calculate all possible radiance bands from a given RGB image.

    Parameters
    ----------
    rgb: xr.DataSet
        Dataset containing an RGB image, exposure and inversion information.

    Returns
    -------
    radiance : xr.Dataset
        Dataset containing radiance images for each passband peak.

    """
    if c.camera_exposure in rgb[c.rgb_data].attrs:
        exposure = rgb[c.rgb_data].attrs[c.camera_exposure]
    else:
        exposure = rgb[c.camera_exposure].values

    radiance = rgb[c.sinv_data].dot(rgb[c.rgb_data]) / exposure
    radiance = radiance.to_dataset(name=c.radiance_data)
    return radiance


def subtract_dark(
        data,
        dark,
        dc_attr=c.dc_included_attr):
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

    """

    if dc_attr in data.attrs and not data.attrs[dc_attr]:
            warn(UserWarning(
                'Data already has {} set to false!'.format(dc_attr)))

    data = xr.apply_ufunc(_subtract_dark, data, dark)
    data.attrs[dc_attr] = False

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
        dims=c.RGB_dims,
        coords={
            c.height_coord: cfa.y,
            c.width_coord: cfa.x,
            c.colour_coord: ['R', 'G', 'B']},
        attrs=cfa.attrs)
