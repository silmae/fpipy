# -*- coding: utf-8 -*-

"""Reading and interpolating raw CFA data.

Example
-------
Loading, converting and plotting data can be done as follows::
    import xarray as xr
    import fpipy as fpi
    import matplotlib


    data = fpi.read_cfa('./data/house_crop_4b_RAW.dat')
    radiance = fpi.raw_to_radiance(data)
    radiance.sel(wavelength=600, method='nearest').plot()
"""

import os
from enum import IntEnum
import xarray as xr
import colour_demosaicing as cdm
from .meta import load_hdt, metalist
from .uglybilinear import demosaicing_CFA_Bayer_uglybilinear


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


def raw_to_radiance(dataset, pattern=None, dm_method='bilinear'):
    """Performs demosaicing and computes radiance from RGB values.

    Parameters
    ----------
    dataset : xarray.Dataset
        Requires data to be found via dataset.cfa, dataset.npeaks,
        dataset.sinvs, dataset.wavelength, dataset.fwhm and dataset.exposure.

    pattern : BayerPattern, optional
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

    pattern_name = BayerPattern(pattern).name

    radiance = {}

    for layer in layers:
        demo = demosaic(layer, pattern_name, dm_method)
        
        for n in range(1, dataset.npeaks.sel(band=layer.band).values + 1):
            data = dataset.sel(band=layer.band, peak=n)
       
            rad = data.sinvs.dot(demo)/data.exposure

            rad.coords['wavelength']=data.wavelength
            rad.coords['fwhm']=data.fwhm
            rad = rad.drop('peak')
            rad = rad.drop('band')
            radiance[float(rad.wavelength)] = rad

    radiance = xr.concat([radiance[key] for key in sorted(radiance)],
                         dim='wavelength',
                         coords=['wavelength', 'fwhm'])
    radiance.attrs = {key: value for key, value in dataset.attrs.items()
                      if key not in ['dark_layer_included', 'bayer_pattern']}

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

def demosaic(cfa, pattern_name, dm_method):
    """Perform demosaicing on a DataArray.

    Parameters
    ----------
    cfa : xarray.DataArray

    pattern_name : str

    dm_method : str

    Returns
    -------
    xarray.DataArray
    """
    dm_methods = {
        'bilinear': cdm.demosaicing_CFA_Bayer_bilinear,
        'Malvar2004': cdm.demosaicing_CFA_Bayer_Malvar2004,
        'Menon2007': cdm.demosaicing_CFA_Bayer_Menon2007,
        'uglybilinear': demosaicing_CFA_Bayer_uglybilinear,
        }
    dm_alg = dm_methods[dm_method]
    
    return xr.DataArray(
    dm_alg(cfa, pattern_name),
    dims=['y', 'x', 'rgb'],
    coords={'y': cfa.y, 'x': cfa.x, 'rgb': ['R', 'G', 'B']},
    attrs=cfa.attrs)
