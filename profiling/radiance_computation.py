from time import time

import fpipy as fp
import fpipy.testing as fpt

size = (64, 1080, 1920)
pattern = 'RGGB'

print('******** Generating test data **************')
t00 = time()
raw = fpt.raw(
        cfa=fpt.cfa(size, pattern, R=1, G=2, B=5),
        dark=fpt.dark(size[1:], 1),
        pattern=pattern,
        exposure=2.0,
        gain=1,
        metas=fpt.metadata(size, (200, 1200)),
        wl_range=(200,1200),
        )
t01 = time()
print(f'Test data generation took {t01 - t00} seconds.')

print('******** Testing sequential computation **************')
t0 = time()
seq_rad = fp.raw_to_radiance(raw)
t_seq = time() - t0
print(f'Total time was {t_seq} seconds.')


print('******** Testing Dask computation **************')
t0 = time()
dask_rad = fp.raw_to_radiance(raw.chunk({'index': 1}))
t_graph = time() - t0
print(f'Graph computation took {t_graph} seconds.')

t0 = time()
dask_rad.compute()
t_compute = time() - t0
t_dask = t_graph + t_compute
print(f'Radiance computation took {t_compute} seconds.')
print(f'Total time was {t_dask} seconds ({t_dask / t_seq} of sequential)')

assert(seq_rad == dask_rad)
