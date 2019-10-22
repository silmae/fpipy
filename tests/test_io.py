from fpipy.io import read_calibration
import xarray as xr

def test_read_calibration():
    cube_expected = xr.open_dataset("C:/MyTemp/fpipy/fpipy/data/read_calibration_expected_result.nc")
    cube = read_calibration("C:/MyTemp/fpipy/fpipy/data/test_calib.txt")
    cube2 = read_calibration("C:/MyTemp/fpipy/fpipy/data/house_calib_seq.txt")
    assert cube == cube_expected
    assert cube2 == cube_expected
    