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


def test_cfa_stack_to_da_required_params():
    with pytest.raises(TypeError, message='') as e:
        fpr.cfa_stack_to_da()
    assert 'missing 3 required keyword-only arguments' in str(e.value)

    with pytest.raises(TypeError) as e:
        fpr.cfa_stack_to_da(cfa=1)
    assert 'missing 2 required keyword-only arguments' in str(e.value)

    with pytest.raises(TypeError) as e:
        fpr.cfa_stack_to_da(cfa=1, pattern='RGGB')
    assert 'missing 1 required keyword-only argument' in str(e.value)


@pytest.mark.parametrize('h, exp_y', [(1, [0.5]), (3, [0.5, 1.5, 2.5])])
@pytest.mark.parametrize('w, exp_x', [(1, [0.5]), (3, [0.5, 1.5, 2.5])])
@pytest.mark.parametrize('n, exp_i', [(1, [0]), (3, [0, 1, 2])])
def test_cfa_stack_to_da_defaults(h, w, n, exp_y, exp_x, exp_i):

    cfa = np.ones((n, h, w))
    da = fpr.cfa_stack_to_da(
            cfa=cfa,
            pattern='RGGB',
            includes_dark_current=True
            )

    assert_array_equal(da.y, exp_y)
    assert_array_equal(da.x, exp_x)
    assert_array_equal(da.index, exp_i)


@pytest.mark.parametrize('h, exp_y', [(1, [41]), (3, [10, 20, 40])])
@pytest.mark.parametrize('w, exp_x', [(1, [42]), (3, [11, 21, 42])])
@pytest.mark.parametrize('n, exp_i', [(1, [43]), (3, [12, 22, 44])])
def test_cfa_stack_to_da_default_overrides(h, w, n, exp_y, exp_x, exp_i):

    cfa = np.ones((n, h, w))
    da = fpr.cfa_stack_to_da(
            cfa=cfa,
            pattern='RGGB',
            includes_dark_current=True,
            x=exp_x, y=exp_y, index=exp_i
            )
    assert_array_equal(da.y, exp_y)
    assert_array_equal(da.x, exp_x)
    assert_array_equal(da.index, exp_i)
