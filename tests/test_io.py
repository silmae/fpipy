from fpipy.io import read_calibration
import xarray as xr


def test_read_calibration_tabs():
    '''
    Tests if the result from reading calibration file with
    tab separation gives the correct result
    '''
    cube_expected = xr.open_dataset(
        "./fpipy/data/read_calibration_expected_result.nc"
        )
    cube = read_calibration(
        "./fpipy/data/house_calib_seq.txt"
        )
    assert cube == cube_expected


def test_read_calibration_mixed_whitespace():
    '''
    Tests if the result from reading calibration file with
    mixed whitespace gives the correct result
    '''
    cube_expected = xr.open_dataset(
        "./fpipy/data/read_calibration_expected_result.nc"
        )
    cube = read_calibration(
        "./fpipy/data/house_calib_seq_mixed_whitespace.txt"
        )
    assert cube == cube_expected
