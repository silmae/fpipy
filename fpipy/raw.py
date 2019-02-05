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


def raw_to_reflectance(raw, whiteraw, dataset=True):
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

    dataset: boolean, optional
        Wether the function should return radiance and reflectance together as
        a Dataset. Default is True.

    Returns
    -------
    reflectance: xarray.Dataset or xarray.DataArray
        Includes computed radiance and reflectance as data variables sorted by
        wavelength or just the reflectance DataArray.
    """
    radiance = raw_to_radiance(raw)
    white = raw_to_radiance(whiteraw)
    return radiance_to_reflectance(radiance, white, dataset=dataset)


def radiance_to_reflectance(radiance, white, dataset=True):
    """Computes reflectance from radiance and a white reference cube. The
    assumptions about when an user wants DataArray and when do they want
    Dataset may have to be looked into in the future.

    Parameters
    ----------
    radiance : xarray.Dataset

    white : xarray.Dataset
        White reference image

    dataset: boolean, optional
        Wether the function should return radiance and reflectance together as
        a Dataset. Default is True.

    Returns
    -------
    reflectance: xarray.Datarray or xarray.Dataset
        Reflectance = Radiance / White_Radiance.
    """
    if hasattr(white, c.cfa_data):
        warn('Converting white from raw to radiance automatically!')
        white = raw_to_radiance(white)

    radiance[c.reflectance_data] = (
            radiance[c.radiance_data] / white[c.radiance_data]
            )

    radiance = radiance.assign_attrs({
        'long_name': 'reflectance',
        'units': '1',
        })
    if dataset is False:
        return radiance[c.reflectance_data]
    else:
        return radiance


def raw_to_radiance(raw, dataset=True, **kwargs):
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

    dataset: boolean, optional
        Wether the function should return radiance and reflectance together as
        a Dataset. Default is True.

    Returns
    -------
    radiances: xarray.Dataset or xarray.DataArray
        Includes computed radiance sorted by wavelength. Passes along relevant
        attributes from input raw.
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

    # Add CF attributes
    radiances = radiances.assign_attrs({
        'long_name': 'radiance per unit wavelength',
        'units': 'W sr-1 m-2 nm-1',
        })
    # Return a dataset and reset all coordinates to variables
    # (keeping only the dimension coordinates)
    radiances = radiances.to_dataset(name=c.radiance_data)
    radiances = radiances.reset_coords()

    if dataset is False:
        return radiances[c.radiance_data]  # We have lost wavelength by now :((
    return radiances


def _raw_to_rad(raw, dark=None, dm_method='bilinear'):
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

    Returns
    -------
    res: xr.Dataset
        Dataset containing radiance data and the relevant metadata.

    """
    return raw.pipe(
                subtract_dark, dark
            ).pipe(
                _raw_to_rgb, dm_method
            ).pipe(
                _rgb_to_rad
            )


def _raw_to_rgb(raw, dm_method):
    """Demosaic a dataset of CFA data.

    Parameters
    ----------
    raw: xr.Dataset
        Dataset containing variable cfa and mosaic pattern information, either
        as a variable or an attribute of the cfa variable.

    Returns
    -------
    res: xr.Dataset
        Dataset containing the demosaiced R, G and B layers as a variable.
    """
    if c.cfa_pattern_data in raw:
        pattern = str(raw[c.cfa_pattern_data].values)
    elif c.genicam_pattern_data in raw:
        pattern = str(raw[c.genicam_pattern_data].values)
    elif c.cfa_pattern_data in raw[c.cfa_data].attrs:
        pattern = str(raw[c.cfa_data].attrs[c.cfa_pattern_data])
    elif c.genicam_pattern_data in raw[c.cfa_data].attrs:
        pattern = str(raw[c.cfa_data].attrs[c.genicam_pattern_data])
    else:
        raise ValueError('Bayer pattern not specified.')

    raw[c.rgb_data] = demosaic(
            raw[c.cfa_data],
            pattern,
            dm_method
            )

    return raw


def _rgb_to_rad(rgb):
    """Calculate all possible radiance bands from a given RGB image.

    Parameters
    ----------
    rgb: xr.DataSet
        Dataset containing as variables RGB image, exposure and radiance
        inversion information.

    Returns
    -------
    radiance: xr.Dataset
        Dataset containing radiances for each passband peak as a variable.

    """

    # Retrieve exposure time
    if c.camera_exposure in rgb:
        exposure = rgb[c.camera_exposure].values
    elif c.genicam_exposure in rgb:  # GenICam uses microseconds
        exposure = rgb[c.genicam_exposure].values * 0.001
    elif c.camera_exposure in rgb[c.rgb_data].attrs:
        exposure = rgb[c.rgb_data].attrs[c.camera_exposure]
    elif c.genicam_exposure in rgb[c.rgb_data].attrs:
        exposure = rgb[c.rgb_data].attrs[c.genicam_exposure] * 0.001
    else:
        raise ValueError('Exposure time not specified.')

    # Preserve metadata by setting them as coordinates
    preserved_data = [
            c.wavelength_data,
            c.fwhm_data,
            c.number_of_peaks
            ]
    rgb = rgb.set_coords([c for c in preserved_data if c in rgb])

    # Select only peaks that have data (as defined by c.number_of_peaks)
    rgb = rgb.sel(
        **{c.peak_coord: rgb[c.peak_coord] <= rgb[c.number_of_peaks]}
        )

    # Compute the inversion to radiance and scale by exposure time
    radiance = rgb[c.sinv_data].dot(rgb[c.rgb_data]) / exposure
    return radiance


def subtract_dark(
        data,
        dark=None,
        dc_attr=c.dc_included_attr):
    """Subtracts dark current reference from image data.

    Subtracts a dark reference frame from all the layers in the given data
    and clamps any negative values in the result to zero. If the input data
    already indicates it has had dark current subtracted (having
    the attribute `fpipy.conventions.dc_included_attr` set to true), it
    simply passes the data through and gives a UserWarning.

    Parameters
    ----------
    data: xarray.DataSet
        Dataset containing the raw images (including dark current).

    dark: array-like, optional
        Dark current reference measurement. If not given, tries to use
        data[c.dark_reference_data].

    dc_attr: str, optional
        Attribute to use for checking whether the data includes dark current,
        and to set False afterwards. Default is
        `fpipy.conventions.dc_included_attr`.

    Returns
    -------
    refarray: xarray.Dataset
        Data from which the dark current reference has been subtracted.
        Negative values are clamped to 0.

    """
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

    return data


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
    pattern = BayerPattern.get(pattern).name

    dm_methods = {
        'bilinear': cdm.demosaicing_CFA_Bayer_bilinear,
        'Malvar2004': cdm.demosaicing_CFA_Bayer_Malvar2004,
        'Menon2007': cdm.demosaicing_CFA_Bayer_Menon2007,
        }
    dm_alg = dm_methods[dm_method]

    res = xr.apply_ufunc(
        dm_alg,
        cfa,
        kwargs=dict(pattern=pattern),
        input_core_dims=[(c.height_coord, c.width_coord)],
        output_core_dims=[(c.RGB_dims)])
    res.coords[c.colour_coord] = ['R', 'G', 'B']
    return res
