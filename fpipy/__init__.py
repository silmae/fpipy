# -*- coding: utf-8 -*-
# flake8: noqa

"""Top-level package for Fabry-Perot Imaging in Python."""

__author__ = """fpipy developers"""
__email__ = 'matti.a.eskelinen@student.jyu.fi'
__version__ = '0.1.0'

from fpipy.meta import load_hdt, image_meta
from fpipy.raw import read_cfa, raw_to_radiance, demosaic, subtract_dark

import os.path as osp

pkg_dir  = osp.abspath(osp.dirname(__file__))
data_dir = osp.join(pkg_dir, 'data')
