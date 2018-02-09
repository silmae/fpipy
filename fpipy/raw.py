# -*- coding: utf-8 -*-

"""Reading and interpolating raw CFA data.

Example
-------
Loading, converting and plotting data can be done as follows::

    data = fpi.read_cfa('cameraoutput.dat')
    radiance = fpi.raw_to_radiance(data)
    radiance.sel(wavelength=800, method='nearest').plot()
"""

import os
from enum import IntEnum
import xarray as xr
import colour_demosaicing as cdm
from .meta import load_hdt, metalist


def read_cfa(filepath):
    """Read a raw CFA datafile and metadata to an xarray Dataset.

    For the fpi sensor in JYU, the metadata in the ENVI datafile is not
    relevant but is preserved as dataset.cfa.attrs just in case.
    Wavelength and fwhm data will be replaced with information from metadata
    and number of layers etc. are omitted as redundant.
    Gain and bayer pattern are assumed to be constant within each file.

    Parameters
    ----------
    filepath : str
        Path to the datafile to be opened, either with or without extension.
        Expects data and metadata to have extensions .dat and .hdt.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset derived from the raw image data and accompanying metadata.
    """

    base = os.path.splitext(filepath)[0]
    datfile = base + '.dat'
    hdtfile = base + '.hdt'

    cfa = xr.open_rasterio(datfile)
    meta = load_hdt(hdtfile)

    if 'fwhm' in cfa.coords:
        cfa = cfa.drop('fwhm')
    if 'wavelength' in cfa.coords:
        cfa = cfa.drop('wavelength')

    npeaks = ('band', metalist(meta, 'npeaks'))
    wavelength = (['band', 'peak'], metalist(meta, 'wavelengths'))
    fwhm = (['band', 'peak'], metalist(meta, 'fwhms'))
    setpoints = (['band', 'setpoint'], metalist(meta, 'setpoints'))
    sinvs = (['band', 'peak', 'rgb'], metalist(meta, 'sinvs'))

    dataset = xr.Dataset(
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

    return dataset


def raw_to_radiance(dataset, pattern=None, demosaic='bilinear'):
    """Performs demosaicing and computes radiance from RGB values.

    Parameters
    ----------
    dataset : xarray.Dataset
        Requires data to be found via dataset.cfa, dataset.npeaks,
        dataset.sinvs, dataset.wavelength, dataset.fwhm and dataset.exposure.

    pattern : int, str or {int: str}
        Default mapping is {0: 'GBRG', 1: 'GRBG', 2: 'BGGR', 3: 'RGGB'}

    demosaic : str
        Default is bilinear. Should match a demosaicing_CFA_Bayer_{string}().

    Returns
    -------
    radiance : xarray.DataArray
        Includes computed radiance sorted by wavelength
        and with x, y, wavelength and fwhm as coordinates.
        Passes along relevant attributes from input dataset.
    """

    if dataset.dark_layer_included:
        layers = substract_dark(dataset.cfa)
    else:
        raise UserWarning('Dark layer is not included in dataset!')

    pattern = bayerpattern(dataset, pattern)

    radiance = []
    for layer in layers:
        # eval() may be slightly ugly, but does what is needed here.
        demo = eval('cdm.demosaicing_CFA_Bayer_'+demosaic+'(layer, pattern)')

        for n in range(1, dataset.npeaks.sel(band=layer.band).values + 1):
            sinvs = dataset.sinvs.sel(band=layer.band, peak=n).values
            # Equals sinvs.sel(band=layer.band, peak=n, rgb='R') * demo[:,:,0]
            rad = (sinvs[0] * demo[:, :, 0]
                   + sinvs[1] * demo[:, :, 1]
                   + sinvs[2] * demo[:, :, 2])
            wavelength = dataset.wavelength.sel(band=layer.band, peak=n)
            fwhm = dataset.fwhm.sel(band=layer.band, peak=n)

            radlayer = xr.DataArray(
                rad,
                coords={'y': layer.y,
                        'x': layer.x,
                        'wavelength': wavelength,
                        'fwhm': fwhm},
                dims=['y', 'x'])
            # This might not be correct for all sensors!
            radlayer = radlayer/dataset.exposure
            radiance.append(radlayer)

    radiance = xr.concat(radiance,
                         dim='wavelength',
                         coords=['wavelength', 'fwhm']
                         ).sortby('wavelength')
    radiance.attrs = {key: value for key, value in dataset.attrs.items()
                      if key not in ['dark_layer_included', 'bayer_pattern']}

    return radiance


def substract_dark(array, dark=None):
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
        return array[1:].astype('float64') - array[0]
    else:
        return array[:].astype('float64') - dark


class BayerPattern(IntEnum):
    """Enumeration of the Bayer Patterns as used by FPI headers."""
    GBRG = 0
    GRBG = 1
    BGGR = 2
    RGGB = 3

    def __str__(self):
        return self.name


def bayerpattern(dataset, pattern=None):
    """Matches input to a string describing a Bayer pattern.

    Parameters
    ----------
    dataset : xr.Dataset

    pattern : int, str or {int: str}

    Returns
    -------
    pattern : str
        See documentation of colour_demosaicing for valid strings.
    """

    case = {0: 'GBRG', 1: 'GRBG', 2: 'BGGR', 3: 'RGGB'}

    if pattern is None:
        return case[dataset.bayer_pattern]

    if type(pattern) is str:
        return pattern

    if type(pattern) is int:
        return case[pattern]

    else:
        raise TypeError('Pattern should be either None or an int or string.')
