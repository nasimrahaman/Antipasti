"""Pad volumes to a given shape."""

def pad(vol, outshape=None):
    outshape = (1024, 1024, 1024) if outshape is None else outshape
    # Take measurements
    sy, sx, sz = vol.shape
    # Get padding target
    ty, tx, tz = outshape
    # Get padding size
    py, px, pz = [((t - s)/2, t - (t - s)/2 - s) for t, s in zip([ty, tx, tz], [sy, sx, sz])]
    # Pad away
    pvol = np.pad(vol, pad_width=(py, px, pz), mode='reflect')
    # Done
    return pvol


def load(path):
    return ndu.fromh5(path, 'data')


def save(vol, path):
    assert '.h5' in path, "File must be .h5 volume."
    ndu.toh5(vol, path.replace('.h5', '-padded.h5'))


if __name__ == '__main__':
    import argparse as ap

    import numpy as np
    import Antipasti.netdatautils as ndu

    parsey = ap.ArgumentParser()
    parsey.add_argument('path', help="Path to the volume.")
    parsey.add_argument('--outshape', help="Output shape as comma separated string.",
                        default=None, type=lambda s: [int(item) for item in s.split(',')])
    args = parsey.parse_args()
    # args = ap.Namespace(path='/mnt/localdata02/nrahaman/Neuro/Datasets/FIBHL/block_40.h5', outshape=None)

    # Load
    vol = load(args.path)
    pvol = pad(vol, args.outshape)
    save(pvol, args.path)