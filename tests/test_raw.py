#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `fpipy.raw` module."""

import pytest

from numpy.testing import assert_array_equal
import numpy as np
import xarray as xr
import xarray.testing as xrt

import fpipy.raw as fpr
import fpipy.conventions as c


def test_read_calibration_matches_ENVI(calib_seq, raw_ENVI):
    for d in calib_seq.dims:
        xrt.assert_identical(calib_seq[d], raw_ENVI[d])

    for v in calib_seq.data_vars:
        xrt.assert_allclose(calib_seq[v], raw_ENVI[v])


def test_read_calibration_format(calib_seq):
    assert type(calib_seq) is xr.Dataset

    # These must be found from each calibration dataset
    dims = [
        c.image_index,
        c.peak_coord,
        c.colour_coord,
        c.setpoint_coord,
        ]
    for d in dims:
        assert d in calib_seq.dims

    variables = [
        c.number_of_peaks,
        c.wavelength_data,
        c.fwhm_data,
        c.setpoint_data,
        c.sinv_data,
        ]
    for v in variables:
        assert v in calib_seq.variables


def test_ENVI_rad_format(rad, rad_ENVI):
    assert type(rad_ENVI) is type(rad)

    for dim in rad.dims:
        assert dim in rad_ENVI.dims
    for coord in rad.coords:
        assert coord in rad_ENVI.coords
    for variable in rad.variables:
        assert variable in rad_ENVI.variables


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
        c.camera_gain,
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
        c.radiance_data,
        c.image_index,
        c.peak_coord,
        c.number_of_peaks,
        c.camera_exposure,
        c.camera_gain,
        c.cfa_pattern_data,
        c.wavelength_data,
        c.sinv_data,
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


def test_subtract_dark_keep_variables(raw):
    variables = [
        c.dark_reference_data,
        ]

    default = fpr.subtract_dark(raw)
    keep_all = fpr.subtract_dark(raw, keep_variables=variables)

    for v in variables:
        assert(v not in default.variables)
        assert(v in keep_all.variables)

        keep_one = fpr.subtract_dark(raw, keep_variables=[v])
        assert(v in keep_one.variables)

        for notv in [var for var in variables if var is not v]:
            assert(notv not in keep_one.variables)


def test_raw_to_radiance_keep_variables(raw):
    variables = [
        c.cfa_data,
        c.dark_reference_data,
        c.rgb_data,
        ]

    default = fpr.raw_to_radiance(raw)
    keep_all = fpr.raw_to_radiance(raw, keep_variables=variables)

    for v in variables:
        assert(v not in default.variables)
        assert(v in keep_all.variables)

        keep_one = fpr.raw_to_radiance(raw, keep_variables=[v])
        assert(v in keep_one.variables)

        for notv in [var for var in variables if var is not v]:
            assert(notv not in keep_one.variables)


def test_raw_to_reflectance_keep_variables(raw):
    variables = [
        c.cfa_data,
        c.dark_reference_data,
        c.rgb_data,
        c.radiance_data
        ]

    default = fpr.raw_to_reflectance(raw, raw)
    keep_all = fpr.raw_to_reflectance(raw, raw, keep_variables=variables)

    for v in variables:
        assert(v not in default.variables)
        assert(v in keep_all.variables)

        keep_one = fpr.raw_to_reflectance(raw, raw, keep_variables=[v])
        assert(v in keep_one.variables)

        for notv in [var for var in variables if var is not v]:
            assert(notv not in keep_one.variables)


def test_radiance_to_reflectance_keep_variables(rad):
    variables = [
        c.radiance_data
        ]

    default = fpr.radiance_to_reflectance(rad, rad)
    keep_all = fpr.radiance_to_reflectance(rad, rad, keep_variables=variables)

    for v in variables:
        assert(v not in default.variables)
        assert(v in keep_all.variables)

        keep_one = fpr.radiance_to_reflectance(rad, rad, keep_variables=[v])
        assert(v in keep_one.variables)

        for notv in [var for var in variables if var is not v]:
            assert(notv not in keep_one.variables)


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


def test_reflectance_is_sensible(raw):
    """Reflectance should be 1 if dataset is used as its own white reference
    unless the reflectance is 0/0 = NaN.
    """
    ref = fpr.raw_to_reflectance(raw, raw, keep_variables=[c.radiance_data])

    target = xr.DataArray(np.ones(ref[c.reflectance_data].shape),
                          dims=ref[c.reflectance_data].dims,
                          coords=ref[c.reflectance_data].coords)
    target.values[ref[c.radiance_data].values == 0] = np.nan

    xrt.assert_equal(ref[c.reflectance_data], target)
# def test_subtract_clip():
#    old =
#    new =
#    assert new.dtype is old.dtype
#    assert np.(new <= old)
#    assert np.all(new >= 0)
#    assert new.shape == old.shape
