# -*- coding: utf-8 -*-
# flake8: noqa

"""Top-level package for Fabry-Perot Imaging in Python."""

__author__ = """fpipy developers"""
__email__ = 'matti.a.eskelinen@student.jyu.fi'
__version__ = '0.1.0'

import fpipy.simulate
import fpipy.data
from fpipy.meta import load_hdt, image_meta
from fpipy.raw import read_cfa, raw_to_radiance, demosaic, subtract_dark, cfa_stack_to_da
