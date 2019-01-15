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
from warnings import warn
from enum import IntEnum
import numpy as np
import xarray as xr
import colour_demosaicing as cdm
from . import conventions as c


def _cfa_dataset(
        *,
        cfa,
        dark=None,
        pattern,
        npeaks,
        wavelength,
        fwhm,
        setpoints,
        sinvs,
        coords=None,
        attrs=None
        ):
    """Combine raw CFA data with metadata into an `xr.Dataset` object.

    Parameters
    ----------
    cfa: array-like
        (n,y,x) array of colour filter array images with labelled dimensions
        index, x and y. If dark is supplied,
        See `cfa_stack_to_da`.

    dark: array-like, optional
        Dark current reference data.

    pattern : BayerPattern or str or array-like
        Bayer filter pattern of the sensor. May be an array if multiple
        patterns are included in the data.

    npeaks: array-like
        Number of separate passbands (peaks) for each index.

    wavelength: array-like
        Wavelengths corresponding to each peak for each index.

    fwhm: array-like
        FWHMs corresponding to each peak for each index.

    setpoints: array-like
        1st setpoints (voltage values sent to the PFPI driver)
        for each index.

    sinvs: array-like
        Inversion coefficients for radiance calculation corresponding
        to each index, peak and colour.

    coords: dict, optional
        Extra coordinates to add to the dataset.

    attrs: dict, optional
        Extra attributes to add to the dataset.

    Returns
    -------
    xr.Dataset
        Dataset with the given data and possible generated coordinates.
    """
    data_vars = {
        c.cfa_data: (c.cfa_dims, cfa),
        c.cfa_pattern_data: (c.image_index, pattern),
        c.number_of_peaks: (c.image_index, npeaks),
        c.wavelength_data: ((c.image_index, c.peak_coord), wavelength),
        c.fwhm_data: ((c.image_index, c.peak_coord), fwhm),
        c.setpoint_data: ((c.image_index, c.setpoint_coord), setpoints),
        c.sinv_data: (c.sinv_dims, sinvs),
        }

    if dark is not None:
        if c.dc_included_attr not in cfa.attrs:
            raise(AttributeError(
                ('CFA data must contain the `{}` attribute if dark '
                 'current reference is supplied')))

        data_vars[c.dark_reference_data] = (c.dark_ref_dims, dark)

    res = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=attrs
        )

    if c.image_index not in res.coords:
        res = res.assign_coords(
                **{c.image_index: np.arange(0, cfa.shape[0])})
    if c.height_coord not in res.coords:
        res = res.assign_coords(
                **{c.height_coord: np.arange(0, cfa.shape[1]) + 0.5})
    if c.width_coord not in res.coords:
        res = res.assign_coords(
                **{c.width_coord: np.arange(0, cfa.shape[2]) + 0.5})

    if c.peak_coord not in res.coords:
        res = res.assign_coords(**{c.peak_coord: [1, 2, 3]})
    if c.setpoint_coord not in res.coords:
        res = res.assign_coords(**{c.setpoint_coord: [1, 2, 3]})
    if c.colour_coord not in res.coords:
        res = res.assign_coords(**{c.colour_coord: ['R', 'G', 'B']})

    return res


def raw_to_reflectance(raw, whiteraw, keep_variables=None):
    """Performs demosaicing and computes radiance from RGB values.

    Parameters
    ----------
    raw : xarray.Dataset
        Requires data that includes cfa, npeaks, sinvs, wavelength, fwhm
        exposure.

    white : xarray.Dataset
        Same as raw but for a cube that describes a white reference target.

    pattern : BayerPattern or str, optional
        Bayer pattern used to demosaic the CFA.
        Can be supplied to override the file metadata value in cases where it
        is missing or incorrect.

    dm_method : str, optional
        **{'bilinear', 'DDFAPD', 'Malvar2004', 'Menon2007'}**
        Demosaicing method. Default is bilinear. See the `colour_demosaicing`
        package for more info on the different methods.

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

    white : xarray.Dataset
        White reference image

    keep_variables: list-like, optional
        List of variables to keep in the result, default None.
        If you wish to keep the intermediate data, pass the relevant
        names from `fpipy.conventions`.

    Returns
    -------
    reflectance: xarray.Dataset
        Reflectance = Radiance / White_Radiance.
    """

    radiance[c.reflectance_data] = (
            radiance[c.radiance_data] / white[c.radiance_data]
            )

    radiance[c.reflectance_data] = radiance[c.reflectance_data].assign_attrs({
        'long_name': 'reflectance',
        'units': '1',
        })

    return _drop_variable(radiance, c.radiance_data, keep_variables)


def raw_to_radiance(raw, **kwargs):
    """Performs demosaicing and computes radiance from RGB values.

    Parameters
    ----------
    raw : xarray.Dataset
        A dataset containing the variables
        cfa, sinvs, wavelength, fwhm and exposure.

    pattern : BayerPattern or str, optional
        Bayer pattern used to demosaic the CFA.
        Can be supplied to override the file metadata value in cases where it
        is missing or incorrect. Default 'RGGB'.

    dm_method : str, optional
        **{'bilinear', 'DDFAPD', 'Malvar2004', 'Menon2007'}**
        Demosaicing method. Default is 'bilinear'. See the `colour_demosaicing`
        package for more info on the different methods.

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
    # Calculate radiances from each mosaic image (see _raw_to_rad)
    radiances = raw.groupby(c.image_index).apply(_raw_to_rad, **kwargs)

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
            **{c.band_index: radiances[c.band_index]}
            )

    # Keep only dimensions as coordinates
    radiances = radiances.reset_coords()
    return radiances


def _raw_to_rad(raw, dark=None, dm_method='bilinear', keep_variables=None):
    """Compute all passband peaks from given raw image data.

    Applies subtract_dark, _raw_to_rgb and _rgb_to_rad
    sequentially to compute radiance from raw image mosaics.

    Parameters
    ----------
    raw : xr.Dataset
        Dataset containing raw CFA data to be passed through
        `subtract_dark`, `_raw_to_rgb` and `_rgb_to_rad`.

    dark : array-like, optional
        Dark reference passed to _subtract_dark. Default None.

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
                subtract_dark, dark, keep_variables
            ).pipe(
                _raw_to_rgb, dm_method, keep_variables
            ).pipe(
                _rgb_to_rad, keep_variables
            )


def _raw_to_rgb(raw, dm_method, keep_variables=None):
    """Demosaic a dataset of CFA data.

    Parameters
    ----------
    raw: xr.Dataset
        Dataset containing variable cfa and mosaic pattern information, either
        as a variable or an attribute of the cfa variable.

    keep_variables: list-like, optional
        List of variables to keep in the result, default None.
        If you wish to keep the raw CFA data, pass a list including
        `fpipy.conventions.cfa_data`.

    Returns
    -------
    res: xr.Dataset
        Dataset containing the demosaiced R, G and B layers as a variable.
    """
    if c.cfa_pattern_data in raw[c.cfa_data].attrs:
        pattern = str(raw[c.cfa_data].attrs[c.cfa_pattern_data])
    else:
        pattern = str(raw[c.cfa_pattern_data].values)

    raw[c.rgb_data] = demosaic(
            raw[c.cfa_data],
            pattern,
            dm_method
            )

    return _drop_variable(raw, c.cfa_data, keep_variables)


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

    # Retrieve exposure time
    if c.camera_exposure in rgb[c.rgb_data].attrs:
        exposure = rgb[c.rgb_data].attrs[c.camera_exposure]
    else:
        exposure = rgb[c.camera_exposure].values

    # Select only peaks that have data (as defined by c.number_of_peaks)
    rgb = rgb.sel(
        **{c.peak_coord: rgb[c.peak_coord] <= rgb[c.number_of_peaks]}
        )

    # Compute the inversion to radiance and scale by exposure time
    rgb[c.radiance_data] = rgb[c.sinv_data].dot(rgb[c.rgb_data]) / exposure

    # Add CF attributes
    rgb[c.radiance_data] = rgb[c.radiance_data].assign_attrs({
        'long_name': 'radiance per unit wavelength',
        'units': 'W sr-1 m-2 nm-1',
        })

    return _drop_variable(rgb, c.rgb_data, keep_variables)


def subtract_dark(
        data,
        dark=None,
        keep_variables=None
        ):
    """Subtracts dark current reference from image data.

    Subtracts a dark reference frame from all the layers in the given data
    and clamps any negative values in the result to zero. If the input data
    already indicates it has had dark current subtracted (having
    the attribute `fpipy.conventions.dc_included_attr` set to false), it
    simply passes the data through and gives a UserWarning. Otherwise,
    dark current is subtracted and the result will have the attribute set to
    false.

    Parameters
    ----------
    data: xarray.DataSet
        Dataset containing the raw images (including dark current).

    dark: array-like, optional
        Dark current reference measurement. If not given, tries to use
        data[c.dark_reference_data].

    keep_variables: list-like, optional
        List of variables to keep in the result, default None.
        If you wish to keep the dark reference data, pass a list including
        `fpipy.conventions.dark_reference`.

    Returns
    -------
    refarray: xarray.Dataset
        Data from which the dark current reference has been subtracted.
        Negative values are clamped to 0.

    """
    dc_attr = c.dc_included_attr
    if (dc_attr in data[c.cfa_data].attrs and
       not data[c.cfa_data].attrs[dc_attr]):
        warn(UserWarning(
            ('Data already has {} set to False,'
             'skipping dark subtraction').format(dc_attr)))
    else:
        if dark is None:
            dark = data[c.dark_reference_data]

        data[c.cfa_data] = xr.apply_ufunc(
                _subtract_clip, data[c.cfa_data], dark
                )
        data[c.cfa_data].attrs[dc_attr] = False

    return _drop_variable(data, c.dark_reference_data, keep_variables)


def _subtract_clip(x, y):
    """Subtract y from x and clip to non-negative values.

    Retains numerical type of x and y without introducing underflows.
    """
    result = (x > y) * (x - y)
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
    cfa: xarray.DataArray

    pattern: BayerPattern or str, optional
        Bayer pattern used to demosaic the CFA.
        Can be supplied to override the file metadata value in cases where it
        is missing or incorrect.

    dm_method: str

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

    res = xr.apply_ufunc(
        dm_alg,
        cfa,
        kwargs=dict(pattern=pattern_name),
        input_core_dims=[(c.height_coord, c.width_coord)],
        output_core_dims=[(c.RGB_dims)])
    res.coords[c.colour_coord] = ['R', 'G', 'B']
    return res


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
