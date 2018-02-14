#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `fpipy` package."""

import pytest


import fpipy as fpi
import xarray as xr
import numpy as np

@pytest.fixture
def sample_raw():
    sample_raw = fpi.read_cfa('WhiteRef_RAW.dat')
    return sample_raw


def test_raw_envi_file_loading(sample_raw):
    assert hasattr(sample_raw, 'wavelength')


def test_substract_dark_rollover(sample_raw):
    assert np.any(fpi.raw.substract_dark(sample_raw.cfa) <= 1.0)

@pytest.fixture
def sample_preprocessed_radiance():
    sample_preprocessed_radiance = xr.open_rasterio('WhiteRef_RAD.dat')
    return sample_preprocessed_radiance

@pytest.fixture
def sample_bilinear_radiance():
    sample_bilinear_radiance = fpi.raw_to_radiance(sample_raw())
    return sample_bilinear_radiance

def test_matches_bilinear_preprocessed_file(sample_preprocessed_radiance,
                                            sample_bilinear_radiance):
    assert pytest.approx(sample_preprocessed_radiance.values,
                  sample_bilinear_radiance.values)
