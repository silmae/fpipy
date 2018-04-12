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


def test_subtract_dark_rollover(raw):
    assert np.all(fpi.raw.subtract_dark(raw.cfa) <= raw.cfa)


def test_radiance_calculation_passes(raw):
    assert type(fpi.raw_to_radiance(raw)) is xr.DataArray