# -*- coding: utf-8 -*-

"""Reading and interpolating raw CFA data.

Example
-------
    raw = fpi.read_cfa('/home/jypehama/work/Piirtoheitin/20180122-133111-Valo-0_RAW.dat')
    raw['cfa'] = raw.cfa.astype('float64') Should this part be inside raw_to_radiance or required from user?
    radiance = raw_to_radiance(raw)
    
"""

import os
import xarray as xr
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
    layers = cfa.band.values - 1

    dataset = xr.Dataset(coords={'peak': [1, 2, 3],
                                 'setpoint': [1, 2, 3],
                                 'rgb': ['R', 'G', 'B']},

                         data_vars={'cfa': cfa.drop('wavelength').drop('fwhm').astype('float64'),
                                    'npeaks': ('band', [meta.getint('Image{}'.format(layer), 'npeaks') for layer in layers]),
                                    'wavelength': (['band', 'peak'], [parsevec(meta.get('Image{}'.format(layer), 'wavelengths')) for layer in layers]),
                                    'fwhm': (['band', 'peak'], [parsevec(meta.get('Image{}'.format(layer), 'fwhms')) for layer in layers]),
                                    'setpoints': (['band', 'setpoint'], [parsevec(meta.get('Image{}'.format(layer), 'setpoints')) for layer in layers]),
                                    'sinvs': (['band', 'peak', 'rgb'], [parsesinvs(meta.get('Image{}'.format(layer), 'sinvs')) for layer in layers])},

                         attrs={'fpi_temperature': meta.getfloat('Header', 'fpi temperature'),
                                'description': meta.get('Header', 'description').strip('"'),
                                'dark_layer_included': meta.getboolean('Header', 'dark layer included'),
                                'gain': meta.getfloat('Image0', 'gain'),
                                'bayer_pattern': meta.getint('Image0', 'bayer pattern')})

    return dataset
