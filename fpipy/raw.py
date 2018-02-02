# -*- coding: utf-8 -*-

"""Reading and interpolating raw CFA data."""

import xarray as xr
from . import meta

def read_xarray(filepath):
    """Read a raw CFA datafile to an xarray DataFrame.
    """

    base = os.path.splitext(filepath)[0]
    datfile = base + '.dat'
    hdtfile = base + '.hdt'

    cfa = xr.open_rasterio(datfile)
    meta = meta.load_hdt(hdtfile)

    return cfa, meta
