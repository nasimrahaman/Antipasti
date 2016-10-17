# Add to path
import sys
import os
sys.path.append(os.path.abspath('{}/../'.format(__file__)))

import numpy as np

from tools import csfeeder, path2dict, ffd
import prepfunctions_cityscapes

import Antipasti.prepkit as pk


__doc__ = "Data Logistics for CityScapes."


def buildpreptrains(prepconfig):

    prepconfig = path2dict(prepconfig)

    # Build prepfunctions
    pf = prepfunctions_cityscapes.prepfunctions()

    # Raw:
    ptX = pk.preptrain([pk.cast()])

    # Labels:
    ptY = pk.preptrain([pk.cast(), lambda batch: np.moveaxis(batch, 3, 1)])

    # Mixed
    ptXY = pk.preptrain([pk.funczip((ptX, ptY))])

    if ffd(prepconfig, 'makewmap', False):
        ptXY.append(pf['wmapmaker']())

    if ffd(prepconfig, 'patchmaker', False):
        ptXY.append(pf['patchmaker'](**prepconfig['patchmaker']))

    if ffd(prepconfig, 'ds', False):
        ptXY.append(pf['ds'](**prepconfig['ds']))

    if ffd(prepconfig, 'patchds', False):
        ptXY.append(pf['patchds'](**prepconfig['patchds']))

    if ffd(prepconfig, 'normalize', False):
        ptXY.append(pk.funczip((pk.cast(),)))
        if ffd(prepconfig, 'vgg', True):
            ptXY.append(pk.funczip((pf['normalize'](np.array([73.15835921, 82.90891754, 72.39239876])),)))
        else:
            ptXY.append(pk.funczip((pk.im2double(8),)))
            ptXY.append(pk.funczip((pk.normalizebatch(),)))

    # Done
    return {'XY': ptXY}


def fetchfeeder(dataconf):
    dataconf = path2dict(dataconf)

    # Get preptrains
    preptrains = buildpreptrains(dataconf['prepconfig'])

    # Make feeders (train and validate)
    trfeeder = csfeeder(feederconfig=dataconf['feederconfig']['training'], preptrain=preptrains['XY'])
    vafeeder = csfeeder(feederconfig=dataconf['feederconfig']['validation'], preptrain=preptrains['XY'])

    # Done!
    return {'training': trfeeder, 'validation': vafeeder}


def test(dataconf):
    import Antipasti.vizkit as vz

    # Convert to dict and make sure it checks out
    dataconf = path2dict(dataconf)
    assert 'plotdir' in dataconf.keys(), "The field 'plotdir' must be provided for printing images."

    print("[+] Building datafeeders...")
    # Make feeder
    feeder = fetchfeeder(dataconf)['training']

    try:
        print("[+] Fetching datafeeders from file...")
        # Fetch from feeder
        batches = feeder.next()

        # Print
        for n, batch in enumerate(batches):
            print("Printing object {} of shape {} to file...".format(n, batch.shape))
            vz.printensor2file(batch, savedir=dataconf['plotdir'], mode='image', nameprefix='N{}--'.format(n))

        print("[+] Done!")

    finally:
        feeder.cleanup()


if __name__ == '__main__':

    import argparse
    parsey = argparse.ArgumentParser()
    parsey.add_argument("dataconf", help="Data Configuration file.")
    # args = parsey.parse_args()
    args = argparse.Namespace(dataconf='/mnt/localdata02/nrahaman/Neuro/ConvNet-Backups/CityScapes/'
                                       'SIERPINSKI-0/Configurations/dataconf.yml')

    # Test
    test(args.dataconf)