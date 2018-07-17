#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `fpipy` package."""

import pytest


import fpipy as fpi
from fpipy.data import house_radiance
import xarray as xr


@pytest.fixture
def rad():
    rad = house_radiance()
    return rad
