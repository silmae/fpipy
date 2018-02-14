#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `fpipy` package."""

import pytest


import fpipy as fpi
import xarray as xr
import numpy as np


@pytest.fixture
def raw():
    raw = fpi.read_cfa('data/house_crop_4b_RAW.dat')
    return raw


def test_raw_envi_file_loading(raw):
    assert hasattr(raw, 'wavelength')


def test_substract_dark_rollover(raw):
    assert np.all(fpi.raw.substract_dark(raw.cfa) <= raw.cfa)


@pytest.fixture
def VTT_radiance():
    VTT_radiance = xr.open_rasterio('data/house_crop_4b_RAD.dat')
    return VTT_radiance


@pytest.fixture
def bilinear_radiance(raw):
    bilinear_radiance = fpi.raw_to_radiance(raw, dm_method='bilinear')
    return bilinear_radiance


def test_matches_bilinear_VTT_file(VTT_radiance,
                                            bilinear_radiance):
    assert np.allclose(VTT_radiance.values, bilinear_radiance.values,
                       rtol=0.275, atol=0.001)
