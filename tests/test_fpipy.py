#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `fpipy` package."""

import pytest


import fpipy as fpi
from fpipy.data import house_raw, house_radiance
import xarray as xr
import numpy as np
from numpy.testing import assert_array_equal


@pytest.fixture
def raw():
    raw = house_raw()
    return raw


@pytest.fixture
def rad():
    rad = house_radiance()
    return rad


def test_raw_envi_file_loading(raw):
    assert hasattr(raw, 'wavelength')


def test_subtract_dark_rollover(raw):
    assert np.all(fpi.raw.subtract_dark(raw.cfa) <= raw.cfa)


def test_radiance_calculation_passes(raw):
    assert type(fpi.raw_to_radiance(raw)) is xr.DataArray


def test_cfa_stack_to_da_required_params():
    with pytest.raises(TypeError, message='') as e:
        da = fpi.raw.cfa_stack_to_da()
    assert 'missing 2 required positional arguments' in str(e.value)

    with pytest.raises(TypeError) as e:
        cfa = np.ones((2,2,2))
        da = fpi.raw.cfa_stack_to_da(cfa)
    assert 'missing 1 required positional argument' in str(e.value)


@pytest.mark.parametrize('h, exp_y', [(1, [0.5]), (3, [0.5, 1.5, 2.5])])
@pytest.mark.parametrize('w, exp_x', [(1, [0.5]), (3, [0.5, 1.5, 2.5])])
@pytest.mark.parametrize('k, exp_i', [(1, [0]), (3, [0, 1, 2])])
def test_cfa_stack_to_da_defaults(h, w, k, exp_y, exp_x, exp_i):

    cfa = np.ones((h, w, k))
    da = fpi.raw.cfa_stack_to_da(cfa, 'RGGB')

    assert_array_equal(da.y, exp_y)
    assert_array_equal(da.x, exp_x)
    assert_array_equal(da.index, exp_i)
