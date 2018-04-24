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
        coords={'band': range(0, bs), 'x': x, 'y': y})

    return cfa
