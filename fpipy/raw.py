# -*- coding: utf-8 -*-

"""Reading and interpolating raw CFA data."""

import xarray as xr


def read_xarray(path):
    """Read a raw CFA datafile to an xarray DataFrame."""

    cfa = xr.open_rasterio(path)
    return cfa
