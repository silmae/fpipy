#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `fpipy.simulate` module."""

from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst
from fpipy.bayer import BayerPattern
from fpipy.simulate import fpi_bandpass_lims, mosaic_transmittances


@given(st.floats(min_value=1.), st.integers(min_value=2))
def test_bandpass_lims_interval(d, n):
    low, high = fpi_bandpass_lims(d, n)
    assert low > 0
    assert low <= high


@given(
    npst.array_shapes(min_dims=2, max_dims=2),
    st.sampled_from(BayerPattern),
    npst.arrays(
        npst.floating_dtypes(),
        npst.array_shapes(min_dims=1, max_dims=1).map(lambda sh: (3, *sh))
        )
    )
def test_mosaic_transmittances_shape(sh, pat, T):
    _, b = T.shape
    res = mosaic_transmittances(sh, pat, T)
    assert (*sh, b) == res.shape
