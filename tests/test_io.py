from fpipy.io import read_calibration
import xarray as xr

def test_read_calibration():
    cube_expected = xr.open_dataset("./fpipy/data/read_calibration_expected_result.nc")
    cube = read_calibration("./fpipy/data/test_calib.txt")
    cube2 = read_calibration("./fpipy/data/house_calib_seq.txt")
    assert cube == cube_expected
    assert cube2 == cube_expected
    