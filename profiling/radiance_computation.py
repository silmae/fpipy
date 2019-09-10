from time import time

import fpipy as fp
import fpipy.testing as fpt


def gen_data(size, pattern):
    print('******** Generating test data **************')
    t00 = time()
    raw = fpt.raw(
            cfa=fpt.cfa(size, pattern, R=1, G=2, B=5),
            dark=fpt.dark(size[1:], 1),
            pattern=pattern,
            exposure=2.0,
            gain=1,
            metas=fpt.metadata(size, (200, 1200)),
            wl_range=(200, 1200),
            pxformat='BayerRG12',
            )

    raw.coords['exposure'] = 2.0
    raw.coords['gain'] = 1
    t01 = time()
    print(f'Test data generation took {t01 - t00} seconds.')
    return raw


def bench_sequential(raw):
    print('******** Testing sequential computation **************')
    t0 = time()
    seq_rad = fp.raw_to_radiance(raw)
    t_seq = time() - t0
    print(f'Total time was {t_seq} seconds.')
    return t_seq, seq_rad


def bench_dask(raw, processes):
    print('******** Testing Dask computation **************')

    # from dask.distributed import Client
    # client = Client(processes=processes)

    t0 = time()
    raw['dn'] = raw.dn.chunk({'index': 1})
    dask_rad = fp.raw_to_radiance(raw)
    dask_rad.radiance.data.visualize('radiance_DAG.png')
    t_graph = time() - t0
    print(f'Graph computation took {t_graph} seconds.')

    t0 = time()
    dask_rad.compute()
    t_compute = time() - t0
    t_dask = t_graph + t_compute
    print(f'Radiance computation took {t_compute} seconds.')
    return t_graph, t_dask, dask_rad


if __name__ == '__main__':

    size = (64, 1080, 1920)
    pattern = 'RGGB'

    raw = gen_data(size, pattern)

    t_seq, _ = bench_sequential(raw)
    t_graph, t_dask, _ = bench_dask(raw, processes=False)

    print(f'Total time was {t_dask} seconds ({t_dask / t_seq} of sequential)')
    # assert(res_seq==res_dask)
