# -*- coding: utf-8 -*-

"""Functions for reading data from various formats.
"""

import xarray as xr
import pandas as pd
import os
import configparser
from .meta import parse_meta_to_ds
from . import conventions as c

def read_calibration(calibfile):
    """Read calibration data from a CSV file and return an `xr.Dataset`."""

    df = pd.read_csv(calibfile, delimiter='\t', index_col='index')

    ds = xr.Dataset()
    ds.coords['index'] = xr.DataArray(df.index, dims=('index'))

    ds['npeaks'] = xr.DataArray(df['Npeaks'], dims=('index'))

    spcols = [col for col in df.columns if 'SP' in col]
    if spcols:
        ds['setpoint'] = xr.DataArray(
            df[spcols],
            dims=('index', 'setpoint_index')
            )
    else:
        raise UserWarning('Setpoint information not found, omitting.')

    wlcols = [col for col in df.columns if 'PeakWL' in col]
    fwhmcols = [col for col in df.columns if 'FWHM' in col]
    sinvcols = [col for col in df.columns if 'Sinv' in col]

    ds.coords['peak_index'] = ('peak_index', [1, 2, 3])


    ds['wavelength'] = xr.DataArray(
        df[wlcols],
        dims=('index', 'peak_index'),
        coords={'index': ds.index, 'peak_index': ds.peak_index}
        )

    ds['fwhm'] = xr.DataArray(
        df[fwhmcols],
        dims=('index', 'peak_index'),
        coords={'index': ds.index, 'peak_index': ds.peak_index}
        )

    ds['sinv'] = xr.concat(
        [xr.DataArray(df[sinvcols[k*3:k*3+3]], dims=('index', 'peak_index'),
            coords={'index': ds.index, 'peak_index': ds.peak_index, 'colour': c})
         for k,c in enumerate('RGB')],
        dim='colour')

    return ds


def read_hdt(hdtfile):
    """Load metadata from a .hdt header file (VTT format)."""

    if not os.path.isfile(hdtfile):
        raise(IOError('Header file {} does not exist'.format(hdtfile)))

    meta = configparser.ConfigParser()
    meta.read(hdtfile)

    return parse_meta_to_ds(meta)


def read_ENVI_cfa(filepath):
    """Read ENVI format CFA data and metadata to an xarray Dataset.

    For the VTT format raw ENVI files the ENVI metadata is superfluous and is
    discarded, with the actual metadata read from the separate VTT header file.
    Wavelength and fwhm data will be replaced with information from metadata
    and number of layers etc. are omitted as redundant.
    Gain and bayer pattern are assumed to be constant within each file.

    Parameters
    ----------
    filepath : str
        Path to the datafile to be opened, either with or without extension.
        Expects data and metadata to have extensions .dat and .hdt.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset derived from the raw image data and accompanying metadata.
        If the ENVI data had an included dark layer, it is separated into
        its own data variable in the dataset.
    """

    base = os.path.splitext(filepath)[0]
    datfile = base + '.dat'
    hdtfile = base + '.hdt'

    envi = xr.open_rasterio(datfile)
    envi.attrs.clear()  # Drop irrelevant attributes

    if 'fwhm' in envi.coords:
        envi = envi.drop('fwhm')
    if 'wavelength' in envi.coords:
        envi = envi.drop('wavelength')

    ds = read_hdt(hdtfile)

    if ds.attrs.pop('dark layer included'):
        dark = xr.DataArray(
                envi.values[0, ::],
                dims=c.dark_ref_dims,
                coords={c.height_coord: envi['y'], c.width_coord: envi['x']},
                name='Dark reference'
                )
        cfa_data = envi.values[1:, ::]
    else:
        dark = None
        cfa_data = envi.values

    ds[c.cfa_data] = (c.cfa_dims, cfa_data)

    if dark is not None:
        # Assume that if a dark reference is included,
        # it has not yet been removed from the data.
        ds[c.cfa_data].attrs[c.dc_included_attr] = True
        ds[c.dark_reference_data] = dark

    return ds
