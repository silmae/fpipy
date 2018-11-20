# -*- coding: utf-8 -*-

"""Functions for reading data from various formats.
"""

import xarray as xr
import pandas as pd
import os
import configparser
from .meta import parse_meta_to_ds
from . import conventions as c


def read_calibration(calibfile, wavelength_unit='nm'):
    """Read a CSV calibration file to a structured dataset.

    Parameters
    ----------
    calibfile : str
        Filepath to the CSV file containing the metadata. The CSV is assumed to
        have the following columns (case-sensitive, in no specific order):
        ['Npeaks', 'SP1', 'SP2', 'SP3', 'PeakWL', 'FWHM', 'Sinv']

    wavelength_unit : str, optional
        Unit of the wavelength data in the calibration file.

    Returns
    -------
    xr.Dataset
        Dataset containing the calibration data in a structured format.

    """

    df = pd.read_csv(calibfile, delimiter='\t', index_col='index')

    ds = xr.Dataset()
    ds.coords[c.image_index] = xr.DataArray(df.index, dims=(c.image_index))

    ds[c.number_of_peaks] = xr.DataArray(df['Npeaks'], dims=(c.image_index))

    spcols = [col for col in df.columns if 'SP' in col]
    if spcols:
        ds[c.setpoint_data] = xr.DataArray(
            df[spcols],
            dims=(c.image_index, c.setpoint_coord)
            )
    else:
        raise UserWarning('Setpoint information not found, omitting.')

    wlcols = [col for col in df.columns if 'PeakWL' in col]
    fwhmcols = [col for col in df.columns if 'FWHM' in col]
    sinvcols = [col for col in df.columns if 'Sinv' in col]

    ds.coords[c.peak_coord] = (c.peak_coord, [1, 2, 3])

    ds[c.wavelength_data] = xr.DataArray(
        df[wlcols],
        dims=(c.image_index, c.peak_coord),
        coords={
            c.image_index: ds[c.image_index],
            c.peak_coord: ds[c.peak_coord]
            },
        attrs={
            'units': wavelength_unit,
            'long_name': 'peak center wavelength',
            'standard_name': 'radiation_wavelength',
            }
        )

    ds[c.fwhm_data] = xr.DataArray(
        df[fwhmcols],
        dims=(c.image_index, c.peak_coord),
        coords={
            c.image_index: ds[c.image_index],
            c.peak_coord: ds[c.peak_coord]
            },
        attrs={
            'units': wavelength_unit,
            'long_name': 'full width at half maximum'
            }
        )

    ds[c.sinv_data] = xr.DataArray(
        df[sinvcols].values.reshape(-1, 3, 3),
        dims=(c.image_index, c.peak_coord, c.colour_coord),
        coords={
            c.image_index: ds[c.image_index],
            c.peak_coord: ds[c.peak_coord],
            c.colour_coord: ['R', 'G', 'B'],
            },
        attrs={
            'long_name': 'dn to pseudoradiance inversion coefficients',
            'units': 'J sr-1 m-2 nm-1',
            }
        )

    return ds


def read_hdt(hdtfile):
    """Load metadata from a .hdt header file (VTT format)."""

    if not os.path.isfile(hdtfile):
        raise(IOError('Header file {} does not exist'.format(hdtfile)))

    meta = configparser.ConfigParser()
    meta.read(hdtfile)

    return parse_meta_to_ds(meta)


def read_ENVI_cfa(filepath, raw_unit='dn'):
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

    raw_unit : str, optional
        Units for the raw data.

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

    envi = xr.open_rasterio(datfile, chunks={})
    #envi.load()
    envi.attrs.clear()  # Drop irrelevant attributes

    if 'fwhm' in envi.coords:
        envi = envi.drop('fwhm')
    if 'wavelength' in envi.coords:
        envi = envi.drop('wavelength')

    ds = read_hdt(hdtfile)

    if ds.attrs.pop('dark layer included'):
        ds[c.dark_reference_data] = xr.DataArray(
                envi.data[0, ::],
                dims=c.dark_ref_dims,
                coords={c.height_coord: envi['y'], c.width_coord: envi['x']},
                attrs={'units': raw_unit}
                )
        ds[c.cfa_data] = (c.cfa_dims, envi.data[1:, ::])
        ds[c.cfa_data].attrs[c.dc_included_attr] = True
    else:
        # Note that we do not no whether or not the data still includes dark
        # current (only that there was no reference).
        ds[c.cfa_data] = (c.cfa_dims, envi.data)

    # Set units for the CFA
    ds[c.cfa_data].attrs['units'] = raw_unit
    return ds
