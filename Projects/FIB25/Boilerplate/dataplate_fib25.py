# Add to path
import sys
import os
sys.path.append(os.path.abspath('{}/../'.format(__file__)))

import prepfunctions_fib25
import tools

import Antipasti.trainkit as tk
import Antipasti.netdatautils as ndu
import Antipasti.netdatakit as ndk
import Antipasti.prepkit as pk


__doc__ = "Data Logistics for CREMI."


def ffd(dictionary, key, default=None):
    if key in dictionary.keys():
        return dictionary[key]
    else:
        return default


def path2dict(path):
    if isinstance(path, str):
        return tk.yaml2dict(path)
    elif isinstance(path, dict):
        return path
    else:
        raise NotImplementedError


def buildpreptrains(prepconfig):
    prepconfig = path2dict(prepconfig)

    # Get prepfunctions
    pf = prepfunctions_fib25.prepfunctions()

    # Build preptrain for raw data
    ptX = pk.preptrain([pk.im2double(nbit=8), pk.cast(), pk.normalizebatch(), pf['time2channel']])

    # Build preptrain for ground truth
    ptY = pk.preptrain([pf['time2channel'], pf['seg2membrane']()] +
                       [pf['disttransform'](gain=prepconfig['edt'])] if prepconfig['edt'] is not None else [] +
                       [pk.cast()])

    # Build preptrain for the zipped XY feeder (start with weightmap maker)
    ptXY = pk.preptrain([pf['wmapmaker']])

    # Build preptrain for the zipped feeder with weights
    ptXYW = pk.preptrain([])

    # Elastic transformations if requested
    if prepconfig['elastic']:
        ptXYW.append(pf['elastictransform'](**prepconfig['elastic-params']))

    # Random rotate if requested
    if prepconfig['random-rotate']:
        ptXYW.append(pf['randomrotate']())

    # Random flip if requested
    if prepconfig['random-flip']:
        ptXYW.append(pf['randomflip']())

    if prepconfig['random-flip-z']:
        ptXYW.append(pf['randomflipz']())

    return {'X': ptX, 'Y': ptY, 'XY': ptXY, 'XYW': ptXYW}


def load(loadconfig):
    loadconfig = path2dict(loadconfig)
    # Load from H5
    datasets = {'raw': ndu.fromh5(loadconfig['raw-path']),
                'gt': ndu.fromh5(loadconfig['gt-path'])}

    return datasets


def fetchfeeder(dataconf):
    dataconf = path2dict(dataconf)

    # Load dataset from file
    datasets = load(dataconf['loadconfig'])

    # Build preptrain
    preptrains = buildpreptrains(dataconf['prepconfig'])

    # Make feeders (only membranes for now)
    # Build ground-truth feeder
    gt = ndk.cargo(data=datasets['gt'],
                   axistags='kij', nhoodsize=dataconf['nhoodsize'], stride=dataconf['stride'],
                   ds=dataconf['ds'], batchsize=dataconf['batchsize'], window=['x', 'x', 'x'],
                   preptrain=preptrains['Y'])

    # Build raw data feeder
    rd = gt.clonecrate(data=datasets['raw'], syncgenerators=True)
    rd.preptrain = preptrains['X']

    # Zip feeders (weightmaps come from wmapmaker in preptrains['XY'])
    zippedfeeder = ndk.feederzip([rd, gt], preptrain=preptrains['XY'])

    # Gate feeder
    feeder = ndk.feedergate(zippedfeeder, tools.skipper, preptrain=preptrains['XYW'])

    return feeder


def test(dataconf):
    import Antipasti.vizkit as vz

    # Convert to dict and make sure it checks out
    dataconf = path2dict(dataconf)
    assert 'plotdir' in dataconf.keys(), "The field 'plotdir' must be provided for printing images."

    print("[+] Building datafeeders...")
    # Make feeder
    feeder = fetchfeeder(dataconf)

    print("[+] Fetching datafeeders from file...")
    # Fetch from feeder
    batches = feeder.next()

    # Print
    for n, batch in enumerate(batches):
        print("Printing object {} of shape {} to file...".format(n, batch.shape))
        vz.printensor2file(batch, savedir=dataconf['plotdir'], mode='image', nameprefix='N{}--'.format(n))

    print("[+] Done!")


if __name__ == '__main__':

    import argparse
    parsey = argparse.ArgumentParser()
    parsey.add_argument("dataconf", help="Data Configuration file.")
    args = parsey.parse_args()

    # Test
    test(args.dataconf)
