#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `fpipy.raw` module."""

import pytest

from numpy.testing import assert_array_equal
import xarray as xr
import xarray.testing as xrt

import fpipy.raw as fpr
import fpipy.conventions as c


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

    # These must be found from each raw dataset for radiance calculations to be
    # possible
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
        c.cfa_pattern_data,
        c.wavelength_data,
        c.sinv_data,
        c.cfa_data
        ]
    for v in variables:
        assert v in raw.variables


def test_raw_dark_inclusion(raw):
    # If a dataset contains data and a dark current reference, it must indicate
    # whether it has been subtracted already

    if (c.cfa_data in raw.variables and
       c.dark_reference_data in raw.variables):
        assert c.dc_included_attr in raw[c.cfa_data].attrs
        assert type(raw[c.cfa_data].attrs[c.dc_included_attr]) is bool


def test_raw_to_radiance_format(raw):
    rad = fpr.raw_to_radiance(raw)
    assert type(rad) is xr.Dataset

    # These should exist in radiances computed from CFA data
    dims = c.radiance_dims
    for d in dims:
        assert d in rad.dims
    variables = [
        c.image_index,
        c.peak_coord,
        c.wavelength_data,
        ]
    for v in variables:
        assert v in rad.variables


def test_raw_to_radiance_correctness(raw, rad):
    expected = rad[c.radiance_data].isel(
            x=slice(1, -2), y=slice(1, -2)
            ).transpose(c.band_index, c.height_coord, c.width_coord)
    actual = fpr.raw_to_radiance(raw)[c.radiance_data].isel(
            x=slice(1, -2), y=slice(1, -2)
            ).transpose(c.band_index, c.height_coord, c.width_coord)
    xrt.assert_equal(expected, actual)


def test_subtract_dark_when_needed(raw):

    old = raw[c.cfa_data]

    if old.attrs[c.dc_included_attr]:
        new = fpr.subtract_dark(raw)[c.cfa_data]
        assert_array_equal(new, fpr._subtract_clip(old, 1))
    else:
        with pytest.warns(
                UserWarning,
                match='has {} set to False'.format(c.dc_included_attr)):
            new = fpr.subtract_dark(raw)[c.cfa_data]
            assert_array_equal(new, old)

    assert new.attrs[c.dc_included_attr] is False

# def test_subtract_clip():
#    old =
#    new =
#    assert new.dtype is old.dtype
#    assert np.(new <= old)
#    assert np.all(new >= 0)
#    assert new.shape == old.shape
