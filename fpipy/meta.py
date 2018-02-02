# -*- coding: utf-8 -*-

"""Metadata parsing."""

import configparser
import numpy as np
import os

def load_hdt(hdtfile):
    """Load metadata from a .hdt header file (VTT format)."""

    if not os.path.isfile(hdtfile):
        raise(IOError('Header file {} does not exist'.format(hdtfile)))

    meta = configparser.ConfigParser()
    meta.read(hdtfile)

    return meta


def image_meta(meta, idx):
    """Parse metadata for a given image (layer) in the FPI data."""

    layer = 'Image{}'.format(idx)
    im_meta = dict()
    im_meta['npeaks'] = meta.getint(layer, 'npeaks')
    im_meta['width'] = meta.getint(layer, 'width')
    im_meta['height'] = meta.getint(layer, 'height')
    im_meta['gain'] = meta.getfloat(layer, 'gain')
    im_meta['bayer pattern'] = meta.getint(layer, 'bayer pattern')
    im_meta['index'] = meta.getint(layer, 'index')
    im_meta['wavelengths'] = parsevec(meta.get(layer, 'wavelengths'))
    im_meta['fwhms'] = parsevec(meta.get(layer, 'fwhms'))
    im_meta['setpoints'] = parsevec(meta.get(layer, 'setpoints'))
    im_meta['sinvs'] = parsevec(meta.get(layer, 'sinvs'))
    return im_meta


def parsevec(s):
    """Parse a vector of floats from a string."""
    return np.fromstring(s.strip('"'), dtype='float', sep=' ')
