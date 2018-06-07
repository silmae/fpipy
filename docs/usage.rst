=====
Usage
=====

From the command line
---------------------

Installing the library using pip installs the following
command line programs:

.. toctree::

    raw2rad

As a library
------------

Loading, converting and plotting data can be done as follows::

    import xarray as xr
    import fpipy as fpi
    import matplotlib
    from fpipy import data_dir
    import os.path as osp


    data = fpi.read_cfa(osp.join(data_dir, 'house_crop_4b_RAW.dat'))
    radiance = fpi.raw_to_radiance(data)
    radiance.sel(wavelength=600, method='nearest').plot()
