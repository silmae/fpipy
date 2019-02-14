#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `fpipy.raw` module."""

import numpy as np
import xarray as xr
import xarray.testing as xrt

import fpipy.raw as fpr
import fpipy.conventions as c
from fpipy.raw import BayerPattern


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


def test_pattern_strings():
    for p in ['gbrg', 'GBRG', 'BayerGB']:
        assert BayerPattern.get(p) is BayerPattern.GBRG
    for p in ['bggr', 'BGGR', 'BayerBG']:
        assert BayerPattern.get(p) is BayerPattern.BGGR
    for p in ['rggb', 'rggb', 'BayerRG']:
        assert BayerPattern.get(p) is BayerPattern.RGGB
    for p in ['grbg', 'GRBG', 'BayerGR']:
        assert BayerPattern.get(p) is BayerPattern.GRBG


def test_genicam_patterns():
    assert BayerPattern.BayerGB is BayerPattern.GBRG
    assert BayerPattern.BayerGR is BayerPattern.GRBG
    assert BayerPattern.BayerBG is BayerPattern.BGGR
    assert BayerPattern.BayerRG is BayerPattern.RGGB


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
    # Demosaicing is actually not interpolation on edges currently
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
        c.cfa_data,
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
        c.dark_corrected_cfa_data,
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


def test_reflectance_is_sensible(rad):
    """Reflectance should be 1 if dataset is used as its own white reference
    unless the reflectance is 0/0 = NaN.
    """
    ref = fpr.radiance_to_reflectance(rad, rad, keep_variables=[c.radiance_data])

    target = xr.DataArray(np.ones(ref[c.reflectance_data].shape),
                          dims=ref[c.reflectance_data].dims,
                          coords=ref[c.reflectance_data].coords)
    target.data[rad[c.radiance_data].values == 0] = np.nan

    xrt.assert_equal(ref[c.reflectance_data], target)
