import pytest
import numpy as np
import xarray as xr
import colour_demosaicing as cdm
try:
    import dask
except ImportError:
    dask = None

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


def wavelengths(b):
    start, end = (400, 1200)
    return np.linspace(start, end, b)


@pytest.fixture
def metas(size):
    idxs = size[0]

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
    b, y, x = size

    pattern = BayerPattern.get(pattern).name
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


@pytest.fixture(params=[1, 0.5])
def exposure(request):
    return request.param


@pytest.fixture
def raw(cfa, dark, pattern, exposure, metas):
    b, y, x = cfa.shape
    sinvs, npeaks, wls = metas
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

    if dask:
        return raw.chunk({c.image_index: 1})
    else:
        return raw


@pytest.fixture
def rad(cfa, dark, exposure, metas):
    k, y, x = cfa.shape
    _, npeaks, _ = metas
    hasdark, _ = dark
    b = np.sum(npeaks)

    if hasdark:
        values = np.array([4, 1, 0, 5, 4, 1], dtype=np.float64)
    else:
        values = np.array([5, 2, 1, 7, 6, 3], dtype=np.float64)

    values = values.reshape(-1, 1, 1)
    values = np.tile(values, (b // 6 + 1, 1, 1))[:b]
    data = np.kron(np.ones((y, x), dtype=np.float64), values) / exposure
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
    if dask:
        return rad.chunk({c.band_index: 1})
    else:
        return rad
