import pytest
import numpy as np
try:
    import dask
except ImportError:
    dask = None

import fpipy.testing as fpt
import fpipy.conventions as c
from fpipy.raw import BayerPattern
from fpipy.data import house_raw, house_radiance, house_calibration


@pytest.fixture(scope="session")
def calib_seq():
    return house_calibration()


@pytest.fixture(scope="session")
def rad_ENVI():
    kwargs = {}
    if dask:
        kwargs.update({'chunks': {'band': 1}})
    return house_radiance(**kwargs)


@pytest.fixture(scope="session")
def raw_ENVI():
    kwargs = {}
    if dask:
        kwargs.update({'chunks': {'band': 1}})
    return house_raw(**kwargs)


@pytest.fixture()
def wl_range():
    return (400, 1200)


@pytest.fixture
def metas(size, wl_range):
    return fpt.metadata(size, wl_range)


@pytest.fixture(
    params=[
        (1, 4, 4),
        (2, 2, 2),
        (3, 2, 2),
        (1, 2, 4),
        ])
def size(request):
    return request.param


@pytest.fixture(
    params=[
        BayerPattern.RGGB,
        BayerPattern.GBRG,
        BayerPattern.GRBG,
        BayerPattern.BGGR,
        ])
def pattern(request):
    return request.param


@pytest.fixture
def cfa(size, pattern):
    return fpt.cfa(size, pattern, R=1, G=2, B=5)


@pytest.fixture(params=[0, 1])
def dark_level(request):
    return request.param


@pytest.fixture
def dark(size, dark_level):
    _, y, x = size
    return fpt.dark((y, x), dark_level)


@pytest.fixture(params=[1, 0.5])
def exposure(request, size):
    return request.param


@pytest.fixture(params=[2])
def gain(request):
    return request.param


@pytest.fixture
def raw(cfa, dark, pattern, exposure, gain, metas, wl_range):
    res = fpt.raw(cfa, dark, pattern, exposure, gain, metas, wl_range)
    if dask:
        return res.chunk()
    else:
        return res


@pytest.fixture
def rad(cfa, dark_level, exposure, metas, wl_range):
    rad = fpt.rad(cfa, dark_level, exposure, metas, wl_range)
    if dask:
        return rad.chunk()
    else:
        return rad
