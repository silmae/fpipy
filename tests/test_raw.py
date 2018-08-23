#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `fpipy.raw` module."""

import pytest


import fpipy.raw as fpr
import numpy as np
from numpy.testing import assert_array_equal
from fpipy.data import house_raw
import fpipy.conventions as c


@pytest.fixture
def raw():
    raw = house_raw()
    return raw


def test_raw_envi_file_loading(raw):
    assert hasattr(raw, 'wavelength')


def test_radiance_calculation_passes(raw):
    import xarray as xr
    assert type(fpr.raw_to_radiance2(raw)) is xr.DataArray


def test_subtract_dark_rollover(raw):
    assert np.all(
            fpr.subtract_dark(
                raw[c.cfa_data],
                raw[c.dark_reference_data])
            <= raw[c.cfa_data])


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
