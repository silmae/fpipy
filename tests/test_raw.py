#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `fpipy.raw` module."""

import pytest

import numpy as np
import xarray as xr
from numpy.testing import assert_array_equal

import fpipy.raw as fpr
from fpipy.data import house_raw
import fpipy.conventions as c


@pytest.fixture
def raw_ENVI():
    return house_raw()


@pytest.fixture
def raw():
    raw = xr.Dataset(
        data_vars={
            c.cfa_data: (
                c.cfa_dims,
                np.kron(
                    np.arange(1, 4, dtype=np.uint16).reshape(3,1,1),
                    np.ones((4,4), dtype=np.uint16))
                ),
            c.dark_reference_data: (
                c.dark_ref_dims,
                np.zeros((4,4), dtype=np.uint16)
                ),
            c.number_of_peaks: (c.image_index, np.array([1, 2, 3])),
            c.sinv_data: (
                 (c.image_index, c.peak_coord, c.colour_coord),
                 np.array([[[0,0,1],
                            [0,0,0],
                            [0,0,0]],
                           [[0,1,0],
                            [0,1,1],
                            [0,0,0]],
                           [[1,0,0],
                            [1,0,1],
                            [1,1,1]]])
                ),
            c.cfa_pattern_attribute: 'RGGB',
            c.camera_exposure: 0.5,
            c.wavelength_data: (
                (c.image_index, c.peak_coord),
                np.array([[100,   0,   0],
                          [200, 300,   0],
                          [400, 500, 600]], dtype=np.float64)
                ),
            },
        coords={
            c.image_index: np.arange(3),
            c.peak_coord: np.array([1, 2, 3]),
            c.colour_coord: ['R', 'G', 'B'],
            }
        )

    return raw

def test_ENVI_raw_format(raw, raw_ENVI):
    assert type(raw_ENVI) is type(raw)

    for dim in raw.dims:
        assert dim in raw_ENVI.dims
    for coord in raw.coords:
        assert coord in raw_ENVI.coords
    for variable in raw.variables:
        assert variable in raw_ENVI.variables


def test_raw_format(raw):
    assert type(raw) is xr.Dataset

    dims = [
        c.peak_coord,
        c.colour_coord,
        *c.cfa_dims,
        ]
    for d in dims:
        assert d in raw.dims

    variables = [
        c.number_of_peaks,
        c.camera_exposure,
        c.cfa_pattern_attribute,
        c.wavelength_data,
#        c.fwhm_data,
        c.sinv_data,
#        c.dark_reference_data,
        c.cfa_data
        ]
    for v in variables:
        assert v in raw.variables


def test_raw_to_radiance_format(raw):
    rad = fpr.raw_to_radiance(raw)
    assert type(rad) is xr.Dataset

    dims = c.radiance_dims
    for d in dims:
        assert d in rad.dims


def test_subtract_dark(raw):
    old = raw[c.cfa_data]
    new = fpr.subtract_dark(raw)[c.cfa_data]
    assert new.dtype is old.dtype
    assert np.all(new <= old)
    assert np.all(new >= 0)
