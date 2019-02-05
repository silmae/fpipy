import pytest

import fpipy.testing as fpt
from fpipy.data import house_raw, house_radiance, house_calibration


@pytest.fixture(scope="session")
def calib_seq():
    return house_calibration()


@pytest.fixture(scope="session")
def rad_ENVI():
    return house_radiance()


@pytest.fixture(scope="session")
def raw_ENVI():
    return house_raw()


@pytest.fixture()
def wl_range():
    return (400, 1200)


@pytest.fixture
def metas(size, wl_range):
    return fpt.metadata(size, wl_range)


@pytest.fixture(
    params=[
        (10, 2, 2),
        (1, 8, 8),
        (1, 12, 8),
        (1, 8, 10),
        (3, 8, 8),
        ])
def size(request):
    return request.param


@pytest.fixture(
    params=[
        'GBRG', 'GRBG', 'BGGR', 'RGGB',
        'BayerGB', 'BayerGR', 'BayerBG', 'BayerRG'
        ])
def pattern(request):
    return request.param


@pytest.fixture
def cfa(size, pattern):
    return fpt.cfa(size, pattern, R=1, G=2, B=5)


@pytest.fixture(
    params=[True, False]
    )
def dark(request, size):
    _, y, x = size
    return request.param, fpt.dark((y, x))


@pytest.fixture(params=[1, 0.5])
def exposure(request):
    return request.param


@pytest.fixture
def raw(cfa, dark, pattern, exposure, metas):
    return fpt.raw(cfa, dark, pattern, exposure, metas)


@pytest.fixture
def rad(cfa, dark, exposure, metas, wl_range):
    return fpt.rad(cfa, dark, exposure, metas, wl_range)
