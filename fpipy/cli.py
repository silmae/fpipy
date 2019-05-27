import sys
import os
import argparse
import fpipy as fp
try:
    import dask
    from dask.diagnostics import ProgressBar
except ImportError:
    dask = None


def save_nc(ds, path):
    """Save dataset as netCDF, overwriting"""
    name, _ = os.path.splitext(path)
    if dask:
        return ds.to_netcdf(name + '.nc', mode='w', compute=False)
    else:
        return ds.to_netcdf(name + '.nc', mode='w')


def writable_dir(dirname):
    """Check if the given directory is valid and open for writing."""
    if not os.path.isdir(dirname):
        raise argparse.ArgumentTypeError(
                '{0} is not a valid path'.format(dirname))
    if os.access(dirname, os.W_OK):
        return dirname
    else:
        raise argparse.ArgumentError(
                '{0} is not writable.'.format(dirname))


def readable_cfa(fname):
    """Check if a given filename points to readable CFA data."""

    bname, ext = os.path.splitext(fname)
    envifiles = [bname + e for e in ('.dat', '.hdt', '.hdr')]
    if (not all(os.path.isfile(f) for f in envifiles)
            and not os.path.isfile(bname + '.nc')):
        raise argparse.ArgumentTypeError(
           '{0} is not a valid CFA file'.format(fname)
        )
    else:
        return fname


OUT_FORMATS = {
        'netCDF': save_nc,
                }


def getparser():

    default_format = 'netCDF'
    default_prefix = 'RAD_'
    default_dir = '.'

    parser = argparse.ArgumentParser(
            description='Convert raw CFA data to radiances.')
    parser.add_argument(
            '-f', '--format',
            metavar='output_format',
            default=default_format,
            choices=OUT_FORMATS.keys(),
            help=('Output file format for the processed radiances, '
                  'default "{0}"'.format(default_format)))
    parser.add_argument(
            '-o', '--odir',
            default=default_dir,
            metavar='output_dir',
            type=writable_dir,
            help=('Directory to write output files to, '
                  'default "{0}"'.format(default_dir)))
    parser.add_argument(
            '-p', '--prefix',
            default=default_prefix,
            metavar='output_prefix',
            type=str,
            help=('Filename prefix to add to output filenames, '
                  'default "{0}"'.format(default_prefix)))
    parser.add_argument(
            'inputs',
            metavar='input file',
            nargs='+',
            type=readable_cfa,
            help=('List of files to process to radiance.'))

    return parser


def raw2rad():
    parser = getparser()
    args = parser.parse_args(sys.argv[1:])

    savecmd = OUT_FORMATS[args.format]
    outdir = args.odir
    inputs = args.inputs
    prefix = args.prefix

    if dask is not None:
        chunking = {'chunks': {'band': 'auto', 'x': 'auto', 'y': 'auto'}}
    else:
        chunking = {}

    print('Calculating radiances from {0} files...'.format(len(inputs)))
    outputs = []
    for rawfile in inputs:
        rad = fp.raw.raw_to_radiance(fp.read_ENVI_cfa(rawfile, **chunking))
        bname = os.path.basename(rawfile)
        outputs.append(savecmd(rad, os.path.join(outdir, prefix + bname)))

    if dask is not None:
        with ProgressBar():
            dask.compute(outputs)
