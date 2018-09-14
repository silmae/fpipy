# -*- coding: utf-8 -*-

"""Metadata parsing."""

import numpy as np
import xarray as xr
from .raw import BayerPattern
from . import conventions as c


def parse_meta_to_ds(meta):
    """Parse FPI metadata from a configparser to a xarray.Dataset."""

    header = meta['Header']
    attrs = {
        'FPI temperature': header.getfloat('FPI Temperature'),
        'dark layer included':
            header.getboolean('Dark Layer included'),
        'description': header['Description'].strip('"')
        }

    layers = meta.sections()
    layers.remove('Header')
    if attrs['dark layer included']:
        layers.remove('Image0')

    ds = xr.concat(
            [parse_image_meta(meta, layer) for layer in layers],
            dim='index'
            )

    ds = ds.assign_attrs(attrs)
    return ds


def parse_image_meta(meta, layer):
    """Parse metadata for a given image (layer) in the FPI data."""

    im_meta = xr.Dataset()
    im_meta[c.number_of_peaks] = meta.getint(layer, 'npeaks')
    im_meta[c.image_width] = meta.getint(layer, 'width')
    im_meta[c.image_height] = meta.getint(layer, 'height')
    im_meta[c.camera_gain] = meta.getfloat(layer, 'gain')
    im_meta[c.camera_exposure] = meta.getfloat(layer, 'exposure time (ms)')
    im_meta[c.cfa_pattern_data] = str(
            BayerPattern(meta.getint(layer, 'bayer pattern')))
    im_meta[c.image_index] = meta.getint(layer, 'index')
    im_meta[c.wavelength_data] = parse_peakmeta(meta.get(layer, 'wavelengths'))
    im_meta[c.fwhm_data] = parse_peakmeta(meta.get(layer, 'fwhms'))
    im_meta[c.setpoint_data] = parse_setpoints(meta.get(layer, 'setpoints'))
    im_meta[c.sinv_data] = parse_sinvs(meta.get(layer, 'sinvs'))

    return im_meta


def parse_peakmeta(s):
    return xr.DataArray(
            parsevec(s),
            dims=(c.peak_coord),
            coords={c.peak_coord: [1, 2, 3]}
            )


def parse_setpoints(s):
    return xr.DataArray(
            parsevec(s),
            dims=(c.setpoint_coord),
            coords={c.setpoint_coord: ['SP1', 'SP2', 'SP3']}
            )


def parse_sinvs(s):
    """Parse an array of floats from a string."""
    vector = parsevec(s)
    return xr.DataArray(
            [vector[0:3], vector[3:6], vector[6:]],
            dims=(c.peak_coord, c.colour_coord),
            coords={
                c.peak_coord: [1, 2, 3],
                c.colour_coord: ['R', 'G', 'B']}
            )


def parsevec(s):
    """Parse a vector of floats from a string."""
    return np.fromstring(s.strip('"'), dtype='float', sep=' ')
