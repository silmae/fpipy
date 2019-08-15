# -*- coding: utf-8 -*-
# flake8: noqa

"""Top-level package for Fabry-Perot Imaging in Python."""

__author__ = """fpipy developers"""
__email__ = 'matti.a.eskelinen@student.jyu.fi'
__version__ = '0.1.0'

__all__ = [
    'simulate',
    'data',
    'meta',
    'read_hdt',
    'read_ENVI_cfa',
    'raw_to_radiance',
    'raw_to_reflectance',
    'radiance_to_reflectance',
    'demosaic',
    'subtract_dark',
]

from . import simulate
from . import data
from . import meta
from . import bayer
from .io import read_hdt, read_ENVI_cfa
from .raw import raw_to_radiance, raw_to_reflectance, radiance_to_reflectance, subtract_dark
