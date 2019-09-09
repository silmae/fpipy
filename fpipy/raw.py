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
from .bayer import rgb_masks_for, inversion_method


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


def raw_to_radiance(ds, keep_variables=None):
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
    ds = ds.copy()
    ds = subtract_dark(ds, keep_variables)

    masks = rgb_masks_for(ds)

    pxformat = ds.get('PixelFormat', None)
    if pxformat is not None:
        pxformat = pxformat.item()
    inv_method = inversion_method(pxformat)

    ds = ds.groupby(c.image_index).apply(
            _raw_to_rad_frame,
            (masks, inv_method, keep_variables)
            )

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

    # Add CF attributes
    ds[c.radiance_data] = ds[c.radiance_data].assign_attrs({
       'long_name': 'radiance per unit wavelength',
       'units': 'W sr-1 m-2 nm-1',
       })

    return ds


def _raw_to_rad_frame(ds, masks, method, keep_variables):

    cfa = ds[c.dark_corrected_cfa_data]
    sinvs = ds[c.sinv_data]
    exposure = find_exposure(ds)

    ds = _drop_variable(ds, c.dark_corrected_cfa_data, keep_variables)
    ds = _drop_variable(ds, c.sinv_data, keep_variables)

    ds[c.radiance_data] = xr.apply_ufunc(
        method,
        cfa, masks, sinvs, exposure,
        input_core_dims=[
            ['y', 'x'], ['colour', 'y', 'x'], ['peak', 'colour'], [],
            ],
        output_core_dims=[['peak', 'y', 'x']],
        dask='parallelized',
        output_dtypes=[np.float64]
    )

    return ds


def find_exposure(ds):
    """Retrieve exposure time in milliseconds from a dataset"""
    if c.camera_exposure in ds:
        exposure = ds[c.camera_exposure].data
    elif c.genicam_exposure in ds:
        # GenICam uses microseconds
        exposure = ds[c.genicam_exposure].data * 0.001
    elif c.camera_exposure in ds[c.ds_data].attrs:
        exposure = ds[c.rgb_data].attrs[c.camera_exposure]
    elif c.genicam_exposure in ds[c.ds_data].attrs:
        # GenICam uses microseconds
        exposure = ds[c.rgb_data].attrs[c.genicam_exposure] * 0.001
    else:
        raise ValueError('Exposure time not specified.')

    return exposure


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
    return (x > y) * (x - y)


def _drop_variable(ds, variable, keep_variables):
    """Drop a given variable from the dataset unless whitelisted.

    Parameters
    ----------

    ds : xr.Dataset
        Dataset to drop variable from.

    variable : str
        Variable name to drop.

    keep_variables : list-like
        Whitelist of variables to keep.

    Returns
    -------
    xr.Dataset
        Original dataset with or without the given variable.
    """
    if not keep_variables or variable not in keep_variables:
        return ds.drop(variable)
    else:
        return ds
