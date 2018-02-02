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

#Parsitaan tähän cfa ja meta yhteen xarrayn tietorakenteeseen ja palautetaan se mielummin

    cfa.attrs.clear()
    for item in list(meta['Header']):
        cfa.attrs[item]=meta['Header'][item]

    return cfa
