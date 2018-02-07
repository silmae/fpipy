# -*- coding: utf-8 -*-

"""Reading and interpolating raw CFA data.

Example
-------
    data = fpi.read_cfa('/home/jypehama/work/Piirtoheitin/20180122-133111-Valo-0_RAW.dat')
    radiance = fpi.raw_to_radiance(raw)
    radiance.isel(wavelength=80).plot()
    
"""

import os
import xarray as xr
import colour_demosaicing as cdm
from .meta import load_hdt, parsevec, parsesinvs

def read_cfa(filepath):
    """Read a raw CFA datafile and metadata to an xarray Dataset.


    For the fpi sensor in JYU, the metadata in the ENVI datafile is not relevant but is preserved as dataset.cfa.attrs just in case.
    Wavelength and fwhm data will be replaced with information from metadata and number of layers etc. are omitted as redundant.
    Gain and bayer pattern are assumed to be constant within each file.

    Parameters
    ----------
    filepath : str
        Path to the datafile to be opened, either with or without extension.
        Expects data to have extension .dat and metatadata to have extension .hdt.

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

    layers = cfa.band.values - 1

    dataset = xr.Dataset(coords={'peak': [1, 2, 3],
                                 'setpoint': [1, 2, 3],
                                 'rgb': ['R', 'G', 'B']},

                         data_vars={'cfa': cfa,
                                    'npeaks': ('band', [meta.getint('Image{}'.format(layer), 'npeaks') for layer in layers]),
                                    'wavelength': (['band', 'peak'], [parsevec(meta.get('Image{}'.format(layer), 'wavelengths')) for layer in layers]),
                                    'fwhm': (['band', 'peak'], [parsevec(meta.get('Image{}'.format(layer), 'fwhms')) for layer in layers]),
                                    'setpoints': (['band', 'setpoint'], [parsevec(meta.get('Image{}'.format(layer), 'setpoints')) for layer in layers]),
                                    'sinvs': (['band', 'peak', 'rgb'], [parsesinvs(meta.get('Image{}'.format(layer), 'sinvs')) for layer in layers])},

                         attrs={'fpi_temperature': meta.getfloat('Header', 'fpi temperature'),
                                'description': meta.get('Header', 'description').strip('"'),
                                'dark_layer_included': meta.getboolean('Header', 'dark layer included'),
                                'gain': meta.getfloat('Image0', 'gain'),
                                'exposure': meta.getfloat('Image0', 'exposure time (ms)'),
                                'bayer_pattern': meta.getint('Image0', 'bayer pattern')})

    return dataset

def raw_to_radiance(dataset, pattern=None, demosaic='bilinear'):
    layers = substract_dark(dataset.cfa)
    pattern = {0: 'GBRG', 1: 'GRBG', 2: 'BGGR', 3: 'RGGB'}
    radiance = []
    for layer in layers:
        demo = cdm.demosaicing_CFA_Bayer_bilinear(layer, pattern=pattern[dataset.bayer_pattern])
        for n in range(1, dataset.npeaks.sel(band=layer.band).values + 1):
            sinvs = dataset.sinvs.sel(band=layer.band, peak=n).values
            radlayer = xr.DataArray(sinvs[0] * demo[:,:,0] + sinvs[1] * demo[:,:,1] + sinvs[2] * demo[:,:,2],
                                    coords={'y': layer.y,
                                            'x': layer.x,
                                            'wavelength': dataset.wavelength.sel(band=layer.band, peak=n),
                                            'fwhm': dataset.fwhm.sel(band=layer.band, peak=n)},
                                    dims=['y', 'x'])
            radlayer = radlayer/dataset.exposure
            radiance.append(radlayer)
    radiance = xr.concat(radiance, dim='wavelength', coords=['wavelength','fwhm']).sortby('wavelength')
    radiance.attrs = {key: value for key, value in dataset.attrs.items() if key not in ['dark_layer_included', 'bayer_pattern']}
    return radiance
    
def substract_dark(array, dark=None):
    """Substracts dark reference from other image layers.

    Parameters
    ----------
    array : xarray.DataArray

    dark : xarray.DataArray
        This is typically included as the first layer of array, but can be overridden

    Returns
    -------
    refarray : xarray.DataArray
        Array of layers from which the dark reference layer has been substracted.
        Resulting array will have dtype float64.
    """

    if dark is None:
        return array[1:].astype('float64') - array[0]
    else:
        return array[:].astype('float64') - dark
