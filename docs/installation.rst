.. highlight:: shell

============
Installation
============

Dependencies
------------
Reading ENVI format files currently uses the rasterio library,
which in turn requires GDAL. For this reason, it is best use
conda with the provided environment file to install the dependencies,
and then install the package (and deps not in conda) using pip:

.. code-block:: console
    $ conda env create -f envs/development.yml
    $ source activate fpipy-dev
    $ pip install .

