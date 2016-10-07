# Add to path
import sys
import os
sys.path.append(os.path.abspath('{}/../'.format(__file__)))

from tools import csfeeder, path2dict, ffd
import prepfunctions_cityscapes

import Antipasti.prepkit as pk


__doc__ = "Data Logistics for CityScapes."


def buildpreptrains(prepconfig):

    prepconfig = path2dict(prepconfig)

    # Build prepfunctions
    pf = prepfunctions_cityscapes.prepfunctions()

    # Raw:
    ptX = pk.preptrain([])

    if ffd(prepconfig, 'normalize', False):
        ptX.append(pk.im2double(8))
        ptX.append(pk.normalizebatch())

    # Labels:
    ptY = pk.preptrain([pk.cast()])

    # Mixed
    ptXY = pk.preptrain([pk.funczip((ptX, ptY))])

    if ffd(prepconfig, 'makewmap', False):
        ptXY.append(pf['wmapmaker'])

    if ffd(prepconfig, 'makepatches', False):
        ptXY.append(pf['patchmaker'](**prepconfig['patchmaker-config']))

    if ffd(prepconfig, 'ds', False):
        ptXY.append(pf['ds'](prepconfig['ds']))

    # Done
    return {'XY': ptXY}


def fetchfeeder(dataconf):
    dataconf = path2dict(dataconf)

    # Get preptrains
    preptrains = buildpreptrains(dataconf['prepconfig'])

    # Make feeder
    feeder = csfeeder(feederconfig=dataconf['feederconfig'], preptrain=preptrains['XY'])

    # Done!
    return feeder


def test(dataconf):
    import Antipasti.vizkit as vz

    # Convert to dict and make sure it checks out
    dataconf = path2dict(dataconf)
    assert 'plotdir' in dataconf.keys(), "The field 'plotdir' must be provided for printing images."

    print("[+] Building datafeeders...")
    # Make feeder
    feeder = fetchfeeder(dataconf)

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
    args = parsey.parse_args()

    # Test
    test(args.dataconf)