=====
Usage
=====

Loading, converting and plotting data can be done as follows::

    import xarray as xr
    import fpipy as fpi
    import matplotlib


    data = fpi.read_cfa('./data/house_crop_4b_RAW.dat')
    radiance = fpi.raw_to_radiance(data)
    radiance.sel(wavelength=600, method='nearest').plot()