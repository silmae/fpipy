.. highlight:: shell

============
Installation
============

---------
Using PIP
---------

For normal use, it is enough to install fpipy using pip from git::

    pip install git+https://github.com/silmae/fpipy.git

Optionally, you may also specify dask as an extra dependency if you wish to
enable parallel computation. Currently this is just for convenience (you will
get the same result if you install dask separately)::

    pip install git+https://github.com/silmae/fpipy.git[dask]

If you need to access ENVI files (such as the included example hyperspectral
datasets) you will need to install rasterio. Specifying ENVI as an extra
dependency will install it if you have the required system libraries for GDAL::

    pip install git+https://github.com/silmae/fpipy.git[ENVI]

However, due to the difficulty of installing the necessary libraries (especially on Windows) it is recommended you use `Conda`_ if you wish to access ENVI files.

.. _Conda: https://www.anaconda.com/

-----------
Using Conda
-----------

Using conda, you can create an environment with rasterio and xarray, then install fpipy using pip::

    conda create -n <env_name> rasterio xarray
    pip install git+https://github.com/silmae/fpipy.git

There are also multiple ready-made environments under the `fpipy/envs` directory which may be used to create suitable conda environments using::

   conda env create -n <env_name> --file fpipy/envs/<env_name>.yml

See :ref:`Contributing` for more info on using development environments.


