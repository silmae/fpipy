# -*- coding: utf-8 -*-

"""Reading and interpolating raw CFA data."""

import os
import xarray as xr
from .meta import load_hdt

def read_cfa(filepath):
    """Read a raw CFA datafile to an xarray DataFrame.
    """

    base = os.path.splitext(filepath)[0]
    datfile = base + '.dat'
    hdtfile = base + '.hdt'

    cfa = xr.open_rasterio(datfile)
    meta = load_hdt(hdtfile)

    #For the fpi sensor in JYU, the metadata in the ENVI datafile is not relevant, and can be fully replaced by metadata from the .hdt file.
    cfa.drop('wavelength')
    cfa.drop('fwhm')
    cfa.attrs.clear()
    cfa.attrs['fpi temperature'] = meta.getfloat('Header', 'fpi temperature')
    cfa.attrs['description'] = meta.get('Header', 'description').strip('"')
    cfa.attrs['dark layer included'] = meta.getboolean('Header', 'dark layer included')
    #cfa.attrs['number of layers'] = meta.getint('Header', 'number of layers') #Redundant

    return cfa
