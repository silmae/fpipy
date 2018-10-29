# -*- coding: utf-8 -*-

""" Test images.
"""

import os.path as osp
from xarray import open_rasterio
from ..io import read_ENVI_cfa
from .. import conventions as c


data_dir = osp.abspath(osp.dirname(__file__))


__all__ = ['data_dir',
           'house_radiance',
           'house_raw',
           ]


def house_raw():
    """Raw images of a house and nearby foliage.

    Raw CFA data from the VTT FPI imager.

    CC0

    Returns
    -------

    house_raw : xarray.Dataset
        (4, 400, 400) cube of CFA images with metadata.

    """

    return read_ENVI_cfa(osp.join(data_dir, 'house_crop_4b_RAW.dat'))


def house_radiance():
    """ Radiance image of a house and foliage.

    Radiances calculated by the VTT software from the `house_raw`
    dataset.

    CC0

    Returns
    -------

    house_raw : xarray.DataArray
        (4, 400, 400) radiance cube with metadata.

    """

    res = open_rasterio(osp.join(data_dir, 'house_crop_4b_RAD.dat'))
    res = res.to_dataset(name=c.radiance_data)
    res = res.reset_coords()
    return res
