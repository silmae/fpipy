# -*- coding: utf-8 -*-

"""Top-level package for Fabry-Perot Imaging in Python."""

__author__ = """Matti A. Eskelinen
                Jyri Hämäläinen"""
__email__ = """matti.a.eskelinen@student.jyu.fi
               jyri.p.hamalainen@gmail.com"""
__version__ = '0.1.0'

from fpipy.meta import load_hdt, image_meta
from fpipy.raw import read_cfa, raw_to_radiance
