#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `fpipy.bayer` module."""

from fpipy.bayer import BayerPattern


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
