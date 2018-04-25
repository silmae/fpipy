# -*- coding: utf-8 -*-

"""Tools for simulating CFA data from radiance cubes.
"""

import xarray as xr
import numpy as np
from colour_demosaicing import masks_CFA_Bayer


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
                'wavelength': rad.wavelength.values[:-1]
                }
            )
        S2 = xr.DataArray(
            np.eye(3),
            dims=('colour', 'wavelength'),
            coords={
                'colour':['R','G','B'],
                'wavelength': rad.wavelength.values[1:]
                }
            )

        S = [S1, S2]

        # Simulate a RGGB pattern CFA
        simulated_raw = create_cfa(rad, S, 'RGGB')

    """

    x, y = rad.x, rad.y

    # Assume that we have rectilinear coordinates
    w, h = x.size, y.size
    bs = len(S)

    # TODO: Add support for arbitrary patterns & colours
    masks = xr.DataArray(
                np.array(masks_CFA_Bayer((h, w), str(pattern))),
                dims=('colour', 'y', 'x'),
                coords={'colour': ['R', 'G', 'B'], 'y': y, 'x': x}
                )

    cfadata = np.zeros((bs, h, w))
    for b in range(0, bs):
        s = S[b]
        for c in s.colour:
            mask = masks.sel(colour=c)
            cfadata[b][mask] = xr.dot(
                s.sel(colour=c),
                rad.sel(wavelength=s.wavelength),
                dims='wavelength'
                ).values[mask]

    cfa = xr.DataArray(
        cfadata,
        dims={'band': range(0, b), 'y': range(0, h), 'x': range(0, w)},
        coords={'band': range(1, bs + 1), 'x': x, 'y': y})

    return cfa


def fpi_triplet(wl, l):
    """Generate a triplet of etalon peaks (wavelengths) given the lowest.

    Parameters:
    -----------
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

        l = \\frac{\\lambda (\\lambda - \\Delta\\lambda)}{\\Delta\\lambda 2 n \\cos(\\theta)}

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
