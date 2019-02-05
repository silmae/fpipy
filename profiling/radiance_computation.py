from time import time

import fpipy as fp
import fpipy.testing as fpt

size = (100, 500, 500)
pattern = 'RGGB'
raw = fpt.raw(
        cfa=fpt.cfa(size, pattern, R=1, G=2, B=5),
        dark=(True, fpt.dark(size[1:])),
        pattern=pattern,
        exposure=2.0,
        metas=fpt.metadata(size, (200, 1200))
        )

t0 = time()
rad = fp.raw_to_radiance(raw)
t1 = time()
print(f'Graph computation took {t1 -t0} seconds.')
t2 = time()
rad.compute()
t3 = time()
print(f'Radiance computation took {t3 - t2} seconds.')
print(f'Total time was {t3 - t0} seconds.')
