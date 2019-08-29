import pytest
try:
    import dask
except ImportError:
    dask = None
import xarray.testing as xrt
import numpy as np
import fpipy.conventions as c
import fpipy.raw as fpr
from fpipy.data import house_raw, house_radiance


rasterio = pytest.importorskip('rasterio')


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


def test_read_calibration_matches_ENVI(calib_seq, raw_ENVI):
    for d in calib_seq.dims:
        xrt.assert_identical(calib_seq[d], raw_ENVI[d])

    for v in calib_seq.data_vars:
        xrt.assert_allclose(calib_seq[v], raw_ENVI[v])


def test_ENVI_rad_format(rad_expected, rad_ENVI):
    assert type(rad_ENVI) is type(rad_expected)

    dims = [
        c.band_index,
        c.width_coord,
        c.height_coord,
        ]

    for dim in dims:
        assert dim in rad_ENVI.dims

    coords = [
        c.band_index,
        c.width_coord,
        c.height_coord,
        ]
    for coord in coords:
        assert coord in rad_ENVI.coords

    variables = [
        c.radiance_data,
        c.wavelength_data,
        c.fwhm_data,
        ]
    for variable in variables:
        assert variable in rad_ENVI.variables


def test_ENVI_raw_format(raw, raw_ENVI):
    assert type(raw_ENVI) is type(raw)

    for dim in raw.dims:
        assert dim in raw_ENVI.dims
    for coord in raw.coords:
        assert coord in raw_ENVI.coords
    variables = [
        c.cfa_data,
        c.dark_reference_data,
        c.sinv_data,
        c.camera_exposure,
        c.camera_gain,
        ]
    for variable in variables:
        assert variable in raw_ENVI.variables


def test_ENVI_raw_to_rad_correspondence(raw_ENVI, rad_ENVI):
    rad_computed = fpr.raw_to_radiance(raw_ENVI)

    for v in [c.band_index, c.wavelength_data, c.fwhm_data]:
        assert np.all(rad_computed[v].values == rad_ENVI[v].values)
