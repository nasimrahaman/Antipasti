"""Console Log to convert to ensemble."""

import numpy as np


def sigm(arr):
    return 1 / (1 + np.exp(-arr))


if __name__ == '__main__':

    import Antipasti.netdatautils as ndu

    # ---- PARAMS ----
    MODE = "TEST"
    SIGMOID = False
    # ---- ----- ----

    # Load
    vol201 = ndu.fromh5('/mnt/localdata02/nrahaman/Neuro/ConvNet-Backups/'
                        'FIB25/SIERPINSKI-WD/Results/{}block_pmap-t201.h5'.format(MODE.lower()))
    vol120 = ndu.fromh5('/mnt/localdata02/nrahaman/Neuro/ConvNet-Backups/'
                        'FIB25/SIERPINSKI-WD/Results/{}block_pmap-t120.h5'.format(MODE.lower()))
    vol012 = ndu.fromh5('/mnt/localdata02/nrahaman/Neuro/ConvNet-Backups/'
                        'FIB25/SIERPINSKI-WD/Results/{}block_pmap-t012.h5'.format(MODE.lower()))

    # Crop
    if MODE == "TRAIN":
        vol201 = vol201[1:-1, 4:-4, 4:-4]
        vol120 = vol120[1:-1, 4:-4, 4:-4]
        vol012 = vol012[1:-1, 4:-4, 4:-4]
    elif MODE == "TEST":
        vol201 = vol201[1:-1, 8:-8, 8:-8]
        vol120 = vol120[1:-1, 8:-8, 8:-8]
        vol012 = vol012[1:-1, 8:-8, 8:-8]
    else:
        raise NotImplementedError

    # Transpose
    vol201t = vol201.transpose((1, 2, 0))
    vol120t = vol120.transpose((2, 0, 1))

    # Average
    vol = np.array([vol012, vol120t, vol201t]).mean(axis=0)

    if SIGMOID:
        vol = sigm(vol)

    # Write
    ndu.toh5(vol, '/mnt/localdata02/nrahaman/Neuro/ConvNet-Backups/'
                  'FIB25/SIERPINSKI-WD/Results/{}block_pmap-ensemble.h5'.format(MODE.lower()))
