# -*- coding: utf-8 -*-
# flake8: noqa

"""Top-level package for Fabry-Perot Imaging in Python."""

__author__ = """fpipy developers"""
__email__ = 'matti.a.eskelinen@student.jyu.fi'
__version__ = '0.1.0'

import fpipy.simulate
import fpipy.data
from fpipy.io import load_hdt, read_cfa
from fpipy.meta import image_meta
from fpipy.raw import raw_to_radiance, demosaic, subtract_dark, cfa_stack_to_da
