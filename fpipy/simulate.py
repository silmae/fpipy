# -*- coding: utf-8 -*-

"""Tools for simulating CFA data from radiance cubes.
"""

import xarray as xr
import numpy as np
from .bayer import BayerPattern, rgb_masks


def fpi_bayer_spectral_signal(T_fpi, Q_eff, T_rgb):
    """Spectral signal for a given FPI gap and order.

    Parameters
    ----------
    T_fpi : array-like
        (3, band) array of FPI transmittances for orders n, n+1 and n+2
        for a given etalon gap length.

    Q_eff : array-like
        (band,) array of quantum efficiencies of the sensor

    T_rgb : array-like
        (3, band) array of tranmittances of the R, G and B pixels.

    Returns
    -------
    S : np.ndarray
        (3, 3) matrix of effective transmittances for the FPI imager.
    """
    return np.einsum('pb,b,cb->pc', T_fpi, Q_eff, T_rgb)


def fpi_bandpass_lims(d, n):
    """Bandpass filter limits for a single order of an FPI at given gap.

    Parameters
    ----------
    d : float
        Gap length of the Fabry-Perot etalon.

    n : int
        The order of the FPI peak included in the limits

    Returns
    -------
    (lmin, lmax) : tuple of float
        Minimum and maximum wavelengths that include the three FPI orders.
    """
    lmax = 2 * d * (1 / n + 1 / (2 * n * (n - 1)))
    lmin = 2 * d * (1 / n - 1 / (2 * n * (n - 1)))
    return (lmin, lmax)


def fpi_bayer_imager(radiance, T_fpi, exposure, T_mosaic, Q_eff, pxformat):
    """Simulate a Fabry-Perot interferometer filtered Bayer sensor image.

    Parameters
    ----------
    radiance : array-like
        (y, x, band) array of radiance values.

    T_fpi : array-like
        (a, b) array of Fabry-Perot interferometer transmittances
        for each band and gap length value a.

    exposure : float
        Exposure (integration time) in milliseconds.

    T_mosaic : array-like
        (y, x, band) array of spectral transmittances for the Bayer mosaic.

    Q_eff : array-like
        Quantum efficiencies of the sensor for each band/wavelength.

    pxformat : str
        Pixel format to discretize result to.

    Result
    ------
    np.ndarray
        (a, y, x) stack of Bayer mosaic images
    """

    res = np.zeros((*radiance.shape[:-1], T_fpi.shape[0]))
    for a, T_gap in enumerate(T_fpi):
        # T_fpi is usually mostly zero, so optimize by indexing
        # the data arrays
        peak_idx = np.nonzero(T_gap)
        res[a, ::] = bayer_sensor(
            T_gap[peak_idx] * radiance[::, peak_idx],
            exposure,
            T_mosaic[::, peak_idx],
            Q_eff[peak_idx],
            pxformat
            )
    return res


def bayer_sensor(radiance, exposure, T_mosaic, Q_eff, pxformat):
    """Simulate a Bayer sensor image.

    Parameters
    ----------
    radiance : array-like
        (y, x, band) array of radiance values.

    exposure : float
        Exposure (integration time) in milliseconds.

    T_mosaic : array-like
        (y, x, band) array of spectral transmittances for the Bayer mosaic.

    Q_eff : array-like
        Quantum efficiencies of the sensor for each band/wavelength.

    pxformat : str
        Pixel format to discretize result to.

    Result
    ------
    np.ndarray
        (y, x) Bayer mosaic
    """
    mosaic_radiance = T_mosaic * radiance
    return mono_sensor(mosaic_radiance, exposure, Q_eff, pxformat)


def mosaic_transmittances(shape, pattern, T_rgb):
    """Transmittances of a Bayer filter mosaic.

    Parameters
    ----------
    shape : pair of int
        (y, x) Shape of the filter array.
    pattern : BayerPattern or str
        The Bayer filter pattern of the array.
    T_rgb : array-like
        (3, b) arrays of transmittances of the R, G and B
        pixels for each band.

    Result
    ------
    np.ndarray
        (y, x, b) array of mosaic responses for each band.
    """
    pattern = BayerPattern.get(pattern).name
    masks = rgb_masks(shape, pattern)

    return np.einsum('cb,cyx->yxb', T_rgb, masks)


def mono_sensor(radiance, exposure, Qeff, pxformat):
    """Simulate a monochromatic sensor image.

    Simulates a linear monochromatic sensor response for a given radiance
    signal and exposure.

    Parameters
    ----------
    radiance : array-like
        (y, x, band) array of radiance values.

    exposure : float
        Exposure (integration time) in milliseconds.

    Q_eff : array-like
        Quantum efficiencies of the sensor for each band/wavelength.

    pxformat : str
        Pixel format to discretize result to.

    Return
    ------
    np.ndarray
        (y, x) monochromatic image.
    """
    res = exposure * np.dot(radiance, Qeff)
    return quantize(res, pxformat)


def quantize(im, pxformat):
    """Quantize a floating-point image to the desired pixel format.

    Simple quantization to maximum levels allowed by the pixel format
    and including the full range of the data.

    Parameters
    ----------
    im : array-like
        Array of floating point values.

    pxformat : str
        Pixel format string (as defined in GenICAM).
    """

    # Bits to use for discretization
    bits = {
        'Mono16': 16,
        'Mono12': 12,
        'BayerRG12': 12,
        'BayerGB12': 12,
        'BayerBG12': 12,
        'BayerGB12': 12,
    }

    return _quantize_mono_uint(im, bits[pxformat])


def _quantize_mono_uint(x, bits):
    """Quantize the given array into bits worth of bins.

    Parameters
    ----------
    x : array-like
        Array of values to be quantized
    bits : int
        Number of bits in the output

    Result
    ------
    np.ndarray
        Array of values between (0, 2**bits - 1) using
        the smallest integer type possible
        (See `np.min_scalar_type` for more info).
    """
    new_max = 2 ** bits - 1
    new_type = np.min_scalar_type(new_max)
    qfac = np.nanmax(x) / new_max

    x = x - np.nanmin(x)
    x = x / qfac

    bins = np.arange(0, 2**bits)
    return np.digitize(x, bins, right=True).astype(new_type)


def create_cfa(rad, S, pattern):
    """Simulate a colour filter array data from radiance data.

    Parameters
    ----------
    rad : xarray.DataArray
        Radiance datacube with wavelength information.

    S : list of xarray.DataArray
        Responses for different colours of the CFA for each wavelength

    pattern : BayerPattern or str
        Bayer pattern for the CFA.

    Returns
    -------
    cfa : `xarray.Dataset`
        CFA images with the given pattern and responses.


    Examples
    --------
    Using a mockup response matrix to create a CFA from radiance::

        import xarray as xr
        import numpy as np
        from fpipy.data import house_radiance
        from fpipy.simulate import create_cfa

        # load example radiance data
        rad = house_radiance()
        rad = rad.swap_dims({'band':'wavelength'})

        # create a mockup response matrix
        S1 = xr.DataArray(
            np.eye(3),
            dims=('colour', 'wavelength'),
            coords={
                'colour':['R','G','B'],
                'wavelength': rad.wavelength.data[:-1]
                }
            )
        S2 = xr.DataArray(
            np.eye(3),
            dims=('colour', 'wavelength'),
            coords={
                'colour':['R','G','B'],
                'wavelength': rad.wavelength.data[1:]
                }
            )

        S = [S1, S2]

        # Simulate a RGGB pattern CFA
        simulated_raw = create_cfa(rad, S, 'RGGB')

    """

    x, y = rad.x, rad.y

    # Assume that we have rectilinear coordinates
    width, height = x.size, y.size
    bands = len(S)

    # TODO: Add support for arbitrary patterns & colours
    masks = xr.DataArray(
                rgb_masks((height, width), str(pattern)),
                dims=('colour', 'y', 'x'),
                coords={'colour': ['R', 'G', 'B'], 'y': y, 'x': x}
                )

    cfadata = np.zeros((bands, height, width))
    for band in range(0, bands):
        s = S[band]
        for c in s.colour:
            mask = masks.sel(colour=c)
            cfadata[band][mask] = xr.dot(
                s.sel(colour=c),
                rad.sel(wavelength=s.wavelength),
                dims='wavelength'
                ).data[mask]

    cfa = xr.DataArray(
        cfadata,
        dims={
            'band': range(0, bands),
            'y': range(0, height),
            'x': range(0, width)
            },
        coords={'band': range(1, bands + 1), 'x': x, 'y': y})

    return cfa


def fpi_transmittance_approx(wl, l, R):
    """Transmittance of the FPI at given wavelength, gap and mirror reflectance

    Uses the approximation given by VTT in Saari et al., 2009,
    with :math:`T = 1 - R` and :math:`\\theta = 0`.

    .. math::

        \\frac{(1 - R)}{1 + R^2 - 2 R
        \\cos\\left(\\frac{4 \\pi d}{\\lambda}\\right)}

    Parameters
    ----------
    wl : np.float64
        Wavelength in chosen units (matching gap length)
    l : np.float64
        Gap length in chosen units (matching wavelength)
    R : np.float64
        Reflectance of the FPI mirrors

    Returns
    -------
    np.float64
        Transmittance of the Fabry-Perot interferometer
    """
    return (1 - R) / (1 + R**2 - 2 * R * np.cos(4 * np.pi * l / wl))


def fpi_transmittance(wl, l, R):
    """Transmittance of the FPI at given wavelength, gap and mirror reflectance

    Parameters
    ----------
    wl : np.float64
        Wavelength in chosen units (matching gap length)
    l : np.float64
        Gap length in chosen units (matching wavelength)
    R : np.float64
        Reflectance of the FPI mirrors

    Returns
    -------
    np.float64
        Transmittance of the Fabry-Perot interferometer
    """
    F = finesse_coefficient(R)
    delta = fpi_phase_difference(wl, l)
    return 1 / (1 + F * np.sin(delta / 2) ** 2)


def finesse_coefficient(R):
    """Finesse coefficient of the etalon

    Parameters
    ----------
    R : np.float64
        Reflectance of the etalon mirrors

    Returns
    -------
    np.float64
        Finesse coefficient of the etalon
    """
    return 4 * R / (1 - R)**2


def fpi_phase_difference(wl, l):
    """Phase difference between pairs of transmitted beams in the FPI

    Parameters
    ----------
    wl : np.float64
        Wavelength of the light in chosen units (matching gap length)
    l : np.float64
        Gap length in chosen units (matching wavelength)

    Returns
    -------
    np.float64
        Phase difference in radians
    """
    n = 1
    theta = 1
    return phase_difference(wl, n, l, theta)


def phase_difference(wl, n, l, theta):
    """Phase difference between pairs of transmitted beams in the etalon

    Parameters
    ----------
    wl : np.float64
        Wavelength of the light in chosen units (matching gap length)
    n : np.float64
        Refractive index of the mirrors
    l : np.float64
        Gap length in chosen units (matching wavelength)
    theta : np.float64
        Angle of the beam entering the etalon in radians

    Returns
    -------
    np.float64
        Phase difference in radians
    """
    return 4 * np.pi * n * l * np.cos(theta) / wl


def fpi_triplet(wl, l):
    """Generate a triplet of etalon peaks (wavelengths) given the lowest.

    Parameters
    ----------
    wl : np.float64
        Lowest wavelength of the triplet.

    l : np.float64
        Gap of the etalon.

    Returns
    -------
    (wl1, wl2, wl3) : tuple of np.float64
        Triplet of consecutive peaks of the Fabry-Perot etalon

    """

    wl2 = wl + fsr_fpi(wl, l)
    wl3 = wl2 + fsr_fpi(wl2, l)

    return wl, wl2, wl3


def fsr_fpi(wl, l):
    """Free spectral range in the FPI.

    Special case of `free_spectral_range` with :math:`n_g = 1` for air
    and collimated light at :math:`\\theta=0`.

    Parameters
    ----------
    wl : np.float64
        Wavelength of the nearest peak.

    l : np.float64
        Gap of the etalon.

    Returns
    -------
    np.float64
        FSR of the FPI at the given values.

    """
    ng = 1.0
    theta = 0.0
    return free_spectral_range(wl, l, ng, theta)


def free_spectral_range(wl, l, ng, theta):
    """Free spectral range of the Fabry-Perot etalon as

    .. math::

        \\Delta\\lambda = \\frac{\\lambda^2}{2 n l \\cos(\\theta)}


    Parameters
    ----------
    wl : np.float64
        Wavelength of the nearest peak.

    l : np.float64
        Gap of the etalon.

    ng : np.float64
        Group refractive index of the media.

    theta : np.float64
        Angle of the ligth entering the etalon.


    Returns
    -------
    fsr : np.float64
        Free spectral range of the Fabry-Perot etalon

    """
    return wl**2 / (2 * ng * l * np.cos(theta) + wl)


def fpi_gap(wl, fsr):
    """Approximate value of the FPI gap.

    Special case of `etalon_gap` with :math:`ng = 1` and :math:`\\theta=0`.

    Parameters
    ----------
    wl : np.float64
        Peak wavelength

    fsr : np.float64
        Free spectral range near the peak.

    Returns
    -------
    l : np.float64
        Approximate gap length of the FPI.

    """
    ng = 1.0
    theta = 0.0
    return etalon_gap(wl, fsr, ng, theta)


def etalon_gap(wl, fsr, ng, theta):
    """Approximate value of the Fabry-Perot etalon gap.

    Approximates the gap length :math:`l` from the formula of the FSR as

    .. math::

        l = \\frac{\\lambda (\\lambda - \\Delta\\lambda)}
                  {\\Delta\\lambda 2 n \\cos(\\theta)}

    Parameters
    ----------
    wl : np.float64
        Peak wavelength

    fsr : np.float64
        Free spectral range near the peak.

    ng : np.float64
        Group refractive index of the media.

    theta : np.float64
        Angle of the light entering the etalon.

    Returns
    -------
    l : np.float64
        Approximate gap length of the Fabry-Perot etalon.

    """

    return wl * (wl - fsr) / (2 * ng * fsr * np.cos(theta))
