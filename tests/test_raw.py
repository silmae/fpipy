#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `fpipy.raw` module."""

import pytest

import numpy as np
from numpy.testing import assert_array_equal
import xarray as xr
import xarray.testing as xrt
import colour_demosaicing as cdm

import fpipy.raw as fpr
from fpipy.data import house_raw
import fpipy.conventions as c


@pytest.fixture(scope="session")
def raw_ENVI():
    return house_raw()


@pytest.fixture
def wavelengths(b):
    start, end = (400, 1200)
    return np.linspace(start, end, b)


@pytest.fixture
def metas(idxs):
    # Number of peaks for each index
    npeaks = np.tile([1, 2, 3], idxs + idxs % 3)[:idxs]

    # distinct sinvs for adjacent indices
    tmp = np.array([[[0, 0, 1],
                     [0, 0, 0],
                     [0, 0, 0]],
                    [[0, 1, 0],
                     [0, 1, 1],
                     [0, 0, 0]],
                    [[1, 0, 0],
                     [1, 0, 1],
                     [1, 1, 0]]])
    sinvs = np.tile(tmp, (idxs // 3 + 1, 1, 1))[:idxs, :, :]

    # Reasonable wavelengths for existing peaks
    mask = np.array([[i < npeaks[n] for i in range(3)] for n in range(idxs)])
    wls = np.zeros((idxs, 3))
    wls.T.flat[mask.T.flatten()] = wavelengths(np.sum(npeaks))
    return sinvs, npeaks, wls


@pytest.fixture(
    params=[
#        (1, 2, 2),
#        (2, 4, 4),
        (3, 8, 8)
        ])
def size(request):
    return request.param


@pytest.fixture(
    params=['GBRG', 'GRBG', 'BGGR', 'RGGB']
    )
def pattern(request):
    return request.param


@pytest.fixture
def cfa(size, pattern):
    b, y, x = size

    masks = cdm.bayer.masks_CFA_Bayer((y, x), pattern)
    cfa = np.zeros(size, dtype=np.uint16)
    cfa[:, masks[0]] = 1  # Red
    cfa[:, masks[1]] = 2  # Green
    cfa[:, masks[2]] = 5  # Blue
    return cfa


@pytest.fixture(
    params=[True, False]
    )
def dark(request, size):
    _, y, x = size
    return request.param, np.ones((y, x), dtype=np.uint16)


@pytest.fixture(params=[1])
def exposure(request):
    return request.param


@pytest.fixture
def raw(cfa, dark, pattern, exposure):
    b, y, x = cfa.shape
    sinvs, npeaks, wls = metas(b)
    dc_included, dref = dark

    data = xr.DataArray(
        cfa,
        dims=c.cfa_dims,
        attrs={c.dc_included_attr: dc_included}
        )
    raw = xr.Dataset(
        data_vars={
            c.cfa_data: data,
            c.dark_reference_data: (c.dark_ref_dims, dref),
            c.number_of_peaks: (c.image_index, npeaks),
            c.sinv_data: (
                (c.image_index, c.peak_coord, c.colour_coord),
                sinvs
                ),
            c.cfa_pattern_data: pattern,
            c.camera_exposure: exposure,
            c.wavelength_data: ((c.image_index, c.peak_coord), wls),
            },
        coords={
            c.image_index: np.arange(b),
            c.peak_coord: np.array([1, 2, 3]),
            c.colour_coord: ['R', 'G', 'B'],
            },
        )

    return raw


@pytest.fixture
def rad(cfa, dark, exposure):
    k, y, x = cfa.shape
    _, npeaks, _ = metas(k)
    hasdark, _ = dark
    b = np.sum(npeaks)

    if hasdark:
        values = np.array([5, 2, 1, 7, 6, 3], dtype=np.float64)
    else:
        values = np.array([4, 1, 0, 5, 4, 1], dtype=np.float64)

    values = values.reshape(-1, 1, 1)
    values = np.tile(values, (b // 6 + 1, 1, 1))[:b] / exposure
    data = np.kron(np.ones((y, x), dtype=np.float64), values)
    wls = wavelengths(b)

    rad = xr.Dataset(
            data_vars={
                c.radiance_data: (c.radiance_dims, data),
                c.wavelength_data: (c.band_index, wls),
                },
            coords={
                c.band_index: np.arange(b),
                }
            )
    return rad


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
        c.image_index,
        c.peak_coord,
        c.wavelength_data,
        ]
    for v in variables:
        assert v in rad.variables


def test_raw_to_radiance_correctness(raw, rad):
    print(raw.cfa.shape)
    print(rad.radiance.shape)
    expected = rad[c.radiance_data].isel(
            x=slice(1, -2), y=slice(1, -2)
            ).transpose(c.band_index, c.height_coord, c.width_coord)
    actual = fpr.raw_to_radiance(raw)[c.radiance_data].isel(
            x=slice(1, -2), y=slice(1, -2)
            ).transpose(c.band_index, c.height_coord, c.width_coord)
    xrt.assert_equal(expected, actual)


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

    assert new.attrs[c.dc_included_attr] == False

#def test_subtract_clip():
#    old =
#    new =
#    assert new.dtype is old.dtype
#    assert np.(new <= old)
#    assert np.all(new >= 0)
#    assert new.shape == old.shape
