import fpipy as fp

rawfile = '/home/maaleske/groupphoto_spec3d/SPEC3D/SPEC3D-groupphoto_RAW.dat'

raw = fp.read_ENVI_cfa(rawfile)
rad = fp.raw_to_radiance(raw)
