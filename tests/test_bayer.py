#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `fpipy.bayer` module."""

import pytest
from fpipy.bayer import (
    BayerPattern,
    demosaic_bilin_12bit,
    demosaic_bilin_12bit_scipy,
    demosaic_bilin_float_scipy,
    rgb_masks,
    mosaic,
    invert_RGB,
    )
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst
import numpy as np


@pytest.fixture(
    params=[
        demosaic_bilin_12bit,
        demosaic_bilin_12bit_scipy,
        demosaic_bilin_float_scipy,
    ])
def demosaic_method(request):
    return request.param


_rgb_shapes = npst.array_shapes(
    min_dims=2, max_dims=2, min_side=2,
    ).map(lambda sh: (3, *sh))


_rgb_images_uint = npst.arrays(
    dtype=npst.unsigned_integer_dtypes(),
    shape=_rgb_shapes,
)


_rgb_images_12bit_centered = npst.arrays(
    dtype=np.uint16,
    shape=_rgb_shapes,
    elements=st.integers(0, 2**12-1).map(lambda x: x << 2)
)


_cfa_images_12bit_centered = npst.arrays(
    dtype=np.uint16,
    shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=2),
    elements=st.integers(0, 2**12-1).map(lambda x: x << 2)
)


_cfa_images_float = npst.arrays(
    dtype=npst.floating_dtypes(),
    shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=2),
)

_sinvs = npst.arrays(
    dtype=npst.floating_dtypes(),
    shape=st.tuples(st.integers(1, 3), st.just(3)))


@given(
    _rgb_images_uint,
    _sinvs,
    st.floats())
def test_rgb_inversion_shape(rgb, sinvs, exp):
    result = invert_RGB(rgb, sinvs, exp)
    np.testing.assert_equal(
        result.shape,
        (sinvs.shape[0], rgb.shape[1], rgb.shape[2]))


@given(
    npst.array_shapes(min_dims=2, max_dims=2),
    st.sampled_from(BayerPattern))
def test_masks_shape(shape, pattern):
    masks = rgb_masks(shape, pattern)
    np.testing.assert_equal(masks.shape, [3, shape[0], shape[1]])


@given(
    npst.array_shapes(min_dims=2, max_dims=2),
    st.sampled_from(BayerPattern))
def test_masks_no_overlap(shape, pattern):
    masks = rgb_masks(shape, pattern)
    sum_of_masks = np.sum(masks, axis=0)
    np.testing.assert_equal(np.ones(shape), sum_of_masks)


@given(
    _rgb_images_uint,
    st.sampled_from(BayerPattern),
    )
def test_mosaic_preserves_dtype(rgb, pattern):
    result = mosaic(rgb, pattern)
    assert result.dtype is rgb.dtype


@given(
    _rgb_images_uint,
    st.sampled_from(BayerPattern),
    )
def test_mosaic_shape(rgb, pattern):
    result = mosaic(rgb, pattern)
    np.testing.assert_equal(result.shape, rgb.shape[1:])


@given(
    cfa=_cfa_images_12bit_centered,
    pattern=st.sampled_from(BayerPattern),
    )
def test_demosaic_mosaic_is_id_12bit(demosaic_method, cfa, pattern):
    masks = rgb_masks(cfa.shape, pattern)
    interp_and_mosaic = mosaic(demosaic_method(cfa, masks), pattern)
    np.testing.assert_equal(cfa, interp_and_mosaic)


@given(
    cfa=_cfa_images_12bit_centered,
    pattern=st.sampled_from(BayerPattern),
    )
def test_demosaic_is_bounded(demosaic_method, cfa, pattern):
    masks = rgb_masks(cfa.shape, pattern)
    interpolant = demosaic_method(cfa, masks)
    for orig, interp in zip(cfa * masks, interpolant):
        assert np.max(orig) >= np.max(interp)
    # TODO: testing meaningful lower bound is hard, since cfa can be zero
    # (and cfa * masks always has zeros)


@given(
    _rgb_images_12bit_centered,
    st.sampled_from(BayerPattern),
    )
def test_demosaic_is_interpolant_12bit(demosaic_method, rgb, pattern):
    masks = rgb_masks(rgb.shape[1:], pattern)
    cfa = mosaic(rgb, pattern)
    result = mosaic(demosaic_method(cfa, masks), pattern)
    np.testing.assert_equal(cfa, result)


@given(
    npst.arrays(
        dtype=npst.integer_dtypes(endianness='='),
        shape=npst.array_shapes(min_dims=2, max_dims=2)),
    st.sampled_from(BayerPattern),
    )
def test_demosaic_12bit_preserves_int_dtype(cfa, pattern):
    masks = rgb_masks(cfa.shape, pattern)
    result = demosaic_bilin_12bit(cfa, masks)
    assert result.dtype is cfa.dtype


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
