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


def test_ENVI_rad_format(rad_expected, rad_ENVI):
    assert type(rad_ENVI) is type(rad_expected)

    dims = [
        c.band_index,
        c.width_coord,
        c.height_coord,
        ]

    for dim in dims:
        assert dim in rad_ENVI.dims

    coords = [
        c.band_index,
        c.width_coord,
        c.height_coord,
        ]
    for coord in coords:
        assert coord in rad_ENVI.coords

    variables = [
        c.radiance_data,
        c.wavelength_data,
        c.fwhm_data,
        ]
    for variable in variables:
        assert variable in rad_ENVI.variables


def test_ENVI_raw_format(raw, raw_ENVI):
    assert type(raw_ENVI) is type(raw)

    for dim in raw.dims:
        assert dim in raw_ENVI.dims
    for coord in raw.coords:
        assert coord in raw_ENVI.coords
    for variable in raw.variables:
        assert variable in raw_ENVI.variables


def test_ENVI_raw_to_rad_correspondence(raw_ENVI, rad_ENVI):
    rad_computed = fpr.raw_to_radiance(raw_ENVI)

    for v in [c.band_index, c.wavelength_data, c.fwhm_data]:
        assert np.all(rad_computed[v].values == rad_ENVI[v].values)


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


def test_raw_to_radiance_format(rad_computed):
    assert type(rad_computed) is xr.Dataset

    # These should exist in radiances computed from CFA data
    dims = c.radiance_dims
    for d in dims:
        assert d in rad_computed.dims

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
        assert v in rad_computed.variables


def test_all_peaks_computed(raw, rad_computed):
    idx_counts = np.bincount(rad_computed[c.image_index])
    for i in raw[c.image_index]:
        assert(
            idx_counts[i] == raw.sel(**{c.image_index: i})[c.number_of_peaks]
            )


def test_all_wavelengths_in_order(raw, rad_computed):
    wls = raw.where(raw[c.wavelength_data] > 0)[c.wavelength_data].values
    expected = np.sort(wls[~np.isnan(wls)].ravel())
    actual = rad_computed[c.wavelength_data].values
    assert np.all(expected == actual)


def test_raw_to_radiance_correctness(rad_expected, rad_computed):
    # Demosaicing is actually not interpolation on edges currently
    expected = rad_expected[c.radiance_data].isel(
                x=slice(1, -2), y=slice(1, -2)
            ).transpose(*c.radiance_dims).compute()
    actual = rad_computed[c.radiance_data].isel(
                x=slice(1, -2), y=slice(1, -2)
            ).transpose(*c.radiance_dims).compute()
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


def test_raw_to_radiance_keep_variables(raw, rad_computed):
    variables = [
        c.cfa_data,
        c.dark_corrected_cfa_data,
        c.dark_reference_data,
        c.rgb_data,
        ]

    default = rad_computed
    keep_all = fpr.raw_to_radiance(raw, keep_variables=variables)

    for v in variables:
        assert(v not in default.variables)
        assert(v in keep_all.variables)

        keep_one = fpr.raw_to_radiance(raw, keep_variables=[v])
        assert(v in keep_one.variables)

        for notv in [var for var in variables if var is not v]:
            assert(notv not in keep_one.variables)


def test_radiance_to_reflectance_keep_variables(rad_expected):
    variables = [
        c.radiance_data
        ]

    default = fpr.radiance_to_reflectance(rad_expected, rad_expected)
    keep_all = fpr.radiance_to_reflectance(
            rad_expected, rad_expected, keep_variables=variables)

    for v in variables:
        assert(v not in default.variables)
        assert(v in keep_all.variables)

        keep_one = fpr.radiance_to_reflectance(
                rad_expected, rad_expected, keep_variables=[v])
        assert(v in keep_one.variables)

        for notv in [var for var in variables if var is not v]:
            assert(notv not in keep_one.variables)


def test_reflectance_is_sensible(rad_expected):
    """Reflectance should be 1 if dataset is used as its own white reference
    except where reflectance is 0 / 0, resulting in NaN.
    """
    actual = fpr.radiance_to_reflectance(rad_expected, rad_expected)

    expected = xr.DataArray(
        np.ones(actual[c.reflectance_data].shape),
        dims=actual[c.reflectance_data].dims,
        coords=actual[c.reflectance_data].coords
        )
    expected.data[rad_expected[c.radiance_data].values == 0] = np.nan

    xrt.assert_equal(actual[c.reflectance_data], expected)
