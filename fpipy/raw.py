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
    rad = fpi.raw_to_radiance(data)
    rad.swap_dims({'band': 'wavelength'}).radiance.sel(wavelength=600,
                                                      method='nearest').plot()
"""
import xarray as xr
import numpy as np
from . import conventions as c
from .utils import _drop_variable
from .bayer import demosaic_bilin, create_mask, _raw_to_rad_frame


def raw_to_reflectance(raw, whiteraw, keep_variables=None):
    """Performs demosaicing and computes radiance from RGB values.

    Parameters
    ----------
    raw : xarray.Dataset
        A dataset containing the following variables:
        `c.cfa_data`,
        `c.dark_reference_data`,
        `c.sinv_data`,
        `c.wavelength_data´,
        `c.fwhm_data`
        `c.camera_exposure`

    white : xarray.Dataset
        Same as raw but for a cube that describes a white reference target.

    keep_variables: list-like, optional
        List of variables to keep in the result, default None.
        If you wish to keep the intermediate data, pass the relevant
        names from `fpipy.conventions`.

    Returns
    -------
    reflectance: xarray.Dataset or xarray.DataArray
        Includes computed radiance and reflectance as data variables sorted by
        wavelength or just the reflectance DataArray.
    """
    radiance = raw_to_radiance(raw, keep_variables=keep_variables)
    white = raw_to_radiance(whiteraw, keep_variables=keep_variables)
    return radiance_to_reflectance(
        radiance, white,
        keep_variables=keep_variables
        )


def radiance_to_reflectance(radiance, white, keep_variables=None):
    """Computes reflectance from radiance and a white reference cube.

    Parameters
    ----------
    radiance : xarray.Dataset
        Dataset containing the image(s) to divide by the references.

    white : xarray.Dataset
        White reference image(s).

    keep_variables: list-like, optional
        List of variables to keep in the result, default None.
        If you wish to keep the intermediate data, pass the relevant
        names from `fpipy.conventions`.

    Returns
    -------
    reflectance: xarray.Dataset
        Dataset containing the reflectance and the original metadata for both
        datasets indexed by measurement type.
    """

    res = xr.concat(
        [radiance, white],
        dim=xr.DataArray(
            ['sample', 'white_reference'],
            dims=(c.measurement_type,),
            name=c.measurement_type,
            ),
        )
    res[c.reflectance_data] = (
            radiance[c.radiance_data] / white[c.radiance_data]
        ).assign_attrs({
            'long_name': 'reflectance',
            'units': '1',
        })

    return _drop_variable(res, c.radiance_data, keep_variables)


def raw_to_radiance_fast(ds, keep_variables=None):
    ds = ds.copy()
    ds = subtract_dark(ds, keep_variables)

    masks = np.rollaxis(create_mask(ds.isel(index=0)).data, -1, 0)

    g_krnl = np.array(
        [[0, 1, 0],
         [1, 4, 1],
         [0, 1, 0]], dtype=np.float64) / 4

    rb_krnl = np.array(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]], dtype=np.float64) / 4

    ds = ds.groupby(c.image_index).apply(
            _raw_to_rad_frame,
            masks=masks, g_krnl=g_krnl, rb_krnl=rb_krnl,
            keep_variables=keep_variables)
    ds = ds.stack(
            **{c.band_index: (c.image_index, c.peak_coord)}
            )

    ds = ds.sel(
        **{c.band_index:
            ds[c.peak_coord] <= ds[c.number_of_peaks]}
        )

    # sort ascending by wavelength
    ds = ds.sortby(c.wavelength_data)

    # replace the multiindex band coordinate with the
    # explicit values (0...nbands)
    ds = ds.reset_index(c.band_index)
    ds = ds.assign_coords(
            **{c.band_index: ds[c.band_index] + 1}
            )

    return ds


def raw_to_radiance(raw, **kwargs):
    """Performs demosaicing and computes radiance from RGB values.

    Parameters
    ----------
    raw : xarray.Dataset
        A dataset containing the following variables:
        `c.sinv_data`,
        `c.wavelength_data´,
        `c.fwhm_data`
        `c.camera_exposure`
        `c.cfa_data`,
        `c.dark_reference_data`,

    keep_variables: list-like, optional
        List of variables to keep in the result, default None.
        If you wish to keep the intermediate data, pass the relevant
        names from `fpipy.conventions`.

    Returns
    -------
    radiances: xarray.Dataset
        Includes computed radiance sorted by wavelength along with original
        metadata.
    """

    # Subtract the dark reference (if needed and/or available)
    radiances = subtract_dark(raw, **kwargs)

    # Calculate radiances from each mosaic image (see _raw_to_rad)
    radiances = radiances.groupby(c.image_index).apply(_raw_to_rad, **kwargs)

    # Create a band coordinate including all possible peaks from each index
    # and then drop any that don't actually have data
    # (defined by c.number_of_peaks)
    radiances = radiances.stack(
            **{c.band_index: (c.image_index, c.peak_coord)}
            )

    radiances = radiances.sel(
        **{c.band_index:
            radiances[c.peak_coord] <= radiances[c.number_of_peaks]}
        )

    # Sort ascending by wavelength
    radiances = radiances.sortby(c.wavelength_data)

    # Replace the MultiIndex band coordinate with the
    # explicit values (0...nbands)
    radiances = radiances.reset_index(c.band_index)
    radiances = radiances.assign_coords(
            **{c.band_index: radiances[c.band_index] + 1}
            )

    return radiances


def _raw_to_rad(raw, keep_variables=None):
    """Compute all passband peaks from given raw image data.

    Applies subtract_dark, _raw_to_rgb and _rgb_to_rad
    sequentially to compute radiance from raw image mosaics.

    Parameters
    ----------
    raw : xr.Dataset
        Dataset containing raw CFA data and the dark reference
        to be passed through `subtract_dark`, `_raw_to_rgb` and `_rgb_to_rad`.

    dm_method : str, optional
        Demosaicing method passed to _rgb_to_rad. Default 'bilinear'.

    keep_variables: list-like, optional
        List of variables to keep in the result, default None.
        If you wish to keep the intermediate data, pass the relevant
        names from `fpipy.conventions`.

    Returns
    -------
    res: xr.Dataset
        Dataset containing radiance data and the relevant metadata.

    """
    return raw.pipe(
                 demosaic_bilin
            ).pipe(
                _rgb_to_rad, keep_variables
            )


def _rgb_to_rad(rgb, keep_variables=None):
    """Calculate all possible radiance bands from a given RGB image.

    Parameters
    ----------
    rgb: xr.DataSet
        Dataset containing as variables RGB image, exposure and radiance
        inversion information.

    keep_variables: list-like, optional
        List of variables to keep in the result, default None.
        If you wish to keep the RGB data, pass a list including
        `fpipy.conventions.rgb_data`.

    Returns
    -------
    radiance: xr.Dataset
        Dataset containing radiances for each passband peak as a variable.

    """

    rgb = rgb.copy()
    # Retrieve exposure time
    if c.camera_exposure in rgb:
        exposure = rgb[c.camera_exposure].data
    elif c.genicam_exposure in rgb:  # GenICam uses microseconds
        exposure = rgb[c.genicam_exposure].data * 0.001
    elif c.camera_exposure in rgb[c.rgb_data].attrs:
        exposure = rgb[c.rgb_data].attrs[c.camera_exposure]
    elif c.genicam_exposure in rgb[c.rgb_data].attrs:
        exposure = rgb[c.rgb_data].attrs[c.genicam_exposure] * 0.001
    else:
        raise ValueError('Exposure time not specified.')

    # Compute the inversion to radiance and scale by exposure time
    rgb[c.radiance_data] = rgb[c.sinv_data].dot(rgb[c.rgb_data] / exposure)

    # Add CF attributes
    rgb[c.radiance_data] = rgb[c.radiance_data].assign_attrs({
       'long_name': 'radiance per unit wavelength',
       'units': 'W sr-1 m-2 nm-1',
       })

    return _drop_variable(rgb, c.rgb_data, keep_variables)


def subtract_dark(ds, keep_variables=None):
    """Subtracts dark current reference from image data.

    Subtracts a dark reference frame from all the layers in the given raw data
    and clamps any negative values in the result to zero. The result is stored
    in the dataset as the variable `c.dark_corrected_cfa_data` which is
    overwritten if it exists.

    Parameters
    ----------
    ds: xarray.DataSet
        Dataset containing the raw images in `fpipy.conventions.cfa_data`
        and the dark current reference measurement as
        `fpipy.conventions.dark_reference_data`.

    keep_variables: list-like, optional
        List of variables to keep in the result, default None.
        If you wish to keep the dark reference data and/or the original raw
        images, pass a list including the variable names.

    Returns
    -------
    xarray.Dataset
        Dataset with the dark corrected data as
        `fpipy.conventions.dark_corrected_cfa_data`

    """
    ds = ds.copy()
    ds[c.dark_corrected_cfa_data] = xr.apply_ufunc(
        _subtract_clip, ds[c.cfa_data], ds[c.dark_reference_data],
        dask='allowed',
        output_dtypes=[
            np.result_type(ds[c.cfa_data], ds[c.dark_reference_data])
            ],
        )

    ds = _drop_variable(ds, c.cfa_data, keep_variables)
    ds = _drop_variable(ds, c.dark_reference_data, keep_variables)
    return ds


def _subtract_clip(x, y):
    """Subtract y from x and clip to non-negative values.

    Retains numerical type of x and y without introducing underflows.
    """
    result = (x > y) * (x - y)
    return result
