# -*- coding: utf-8 -*-

"""Tools for simulating CFA data from radiance cubes.
"""

import numpy as np
from .bayer import _bayer_masks


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


def mosaic(rgb, pattern):
    """Create a Bayer filter mosaic from an RGB image.

    Parameters
    ----------
    rgb : xr.DataArray
        (y, x, 3) RGB image array.

    pattern: BayerPattern or str
        Bayer pattern for the mosaic.

    Returns
    -------
    mosaic : np.ndarray
        (y, x) mosaic image.
    """
    masks, _ = _bayer_masks(rgb.shape[:-1], pattern)
    split = masks * np.moveaxis(rgb, -1, 0)
    return np.sum(split, axis=0)
