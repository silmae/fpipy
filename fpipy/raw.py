# -*- coding: utf-8 -*-

"""Reading and interpolating raw CFA data."""

import os
import xarray as xr
from .meta import load_hdt, parsevec, parsesinvs

def read_cfa(filepath):
    """Read a raw CFA datafile and metadata to an xarray Dataset.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    dataset : xarray.Dataset
    """

    base = os.path.splitext(filepath)[0]
    datfile = base + '.dat'
    hdtfile = base + '.hdt'

    cfa = xr.open_rasterio(datfile)
    meta = load_hdt(hdtfile)
    layers = cfa.band.values - 1

    #For the fpi sensor in JYU, the metadata in the ENVI datafile is not relevant but is preserved as dataset.cfa.attrs just in case.
    dataset = xr.Dataset(coords={'peak': [1, 2, 3], 'setpoint': [1, 2, 3], 'rgb': ['R', 'G', 'B']},
                         data_vars={'cfa': cfa.drop('wavelength').drop('fwhm'), #Wavelength and fwhm data will be replaced with information from metadata.
                                    'npeaks': ('band', [meta.getint('Image{}'.format(layer), 'npeaks') for layer in layers]),
                                    'wavelength': (['band', 'peak'], [parsevec(meta.get('Image{}'.format(layer), 'wavelengths')) for layer in layers]),
                                    'fwhm': (['band', 'peak'], [parsevec(meta.get('Image{}'.format(layer), 'fwhms')) for layer in layers]),
                                    'setpoints': (['band', 'setpoint'], [parsevec(meta.get('Image{}'.format(layer), 'setpoints')) for layer in layers]),
                                    'sinvs': (['band', 'peak', 'rgb'], [parsesinvs(meta.get('Image{}'.format(layer), 'sinvs')) for layer in layers])},
                         attrs={'fpi_temperature': meta.getfloat('Header', 'fpi temperature'),
                                'description': meta.get('Header', 'description').strip('"'),
                                'dark_layer_included': meta.getboolean('Header', 'dark layer included'), #Number of layers etc. are omitted as redundant metadata.
                                'gain': meta.getfloat('Image0', 'gain'), #Assumed to be constant for each separate datafile.
                                'bayer_pattern': meta.getint('Image0', 'bayer pattern')}) #Assumed to be constant for each separate datafile.

    return dataset
