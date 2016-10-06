"""Transpose volumes."""


def tp(vol, transpose=None):
    transpose = (0, 1, 2) if transpose is None else transpose
    # Pad away
    tvol = vol.transpose(transpose)
    # Done
    return tvol


def load(path):
    return ndu.fromh5(path, 'data')


def save(vol, path, nameflag=''):
    assert '.h5' in path, "File must be .h5 volume."
    ndu.toh5(vol, path.replace('.h5', '-t{}.h5'.format(nameflag)))


if __name__ == '__main__':
    import argparse as ap
    import Antipasti.netdatautils as ndu

    parsey = ap.ArgumentParser()
    parsey.add_argument('path', help="Path to the volume.")
    parsey.add_argument('--transpose', help="Transpose as a string (e.g. '120' or '201').",
                        default=None, type=lambda s: tuple(int(l) for l in list(s)))
    args = parsey.parse_args()
    # args = ap.Namespace(path='/mnt/localdata02/nrahaman/Neuro/Datasets/FIBHL/block_40.h5', outshape=None)

    # Load
    vol = load(args.path)
    pvol = tp(vol, args.transpose)
    save(pvol, args.path, nameflag=''.join([str(l) for l in args.transpose]))
