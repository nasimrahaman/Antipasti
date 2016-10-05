# Add to path
import sys
import os
sys.path.append(os.path.abspath('{}/../'.format(__file__)))

import prepfunctions_hh
import tools

import Antipasti.trainkit as tk
import Antipasti.netdatautils as ndu
import Antipasti.netdatakit as ndk
import Antipasti.prepkit as pk

import numpy as np

__doc__ = "Data Logistics for HH."


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

    # This is how the iterator would look like:
    # X --- pX ---
    #             \              X
    #              > --- pXY --- | --- XYW --->
    #             /              Y
    # Y --- pY ---

    # Get prepfunctions
    pf = prepfunctions_hh.prepfunctions()

    # pX
    ptpX = pk.preptrain([pk.im2double(nbit=8), pf['time2channel']])

    # pY
    ptpY = pk.preptrain([pf['time2channel']])

    # Build preptrain for the zipped XY feeder (start with weightmap maker)
    ptpXY = pk.preptrain(([pf['wmapmaker']()] if prepconfig['makewmap'] else []))

    # Build preptrain for raw data (X)
    ptX = pk.preptrain([pk.normalizebatch()])

    # Build preptrain for ground truth (Y)
    ptY = pk.preptrain([pf['seg2membrane']()] +
                       [pf['disttransform'](gain=prepconfig['edt'])] if prepconfig['edt'] is not None else [] +
                       [pk.cast()])

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

    return {'X': ptX, 'Y': ptY, 'pXY': ptpXY, 'XYW': ptXYW, 'pX': ptpX, 'pY': ptpY}


def load(loadconfig):
    loadconfig = path2dict(loadconfig)
    # Load from H5
    datasets = {'raw': ndu.fromh5(loadconfig['raw-path'], 'data'),
                'gt': ndu.fromh5(loadconfig['gt-path'], 'data')}

    if ffd(loadconfig, 'transpose'):
        transpose = eval(loadconfig['transpose']) if isinstance(loadconfig['transpose'], str) else (2, 1, 0)
        datasets = {key: dset.transpose(transpose) for key, dset in datasets.items()}

    if ffd(loadconfig, 'pad') is not None:
        datasets = {key: np.lib.pad(dset, tuple(eval(loadconfig['pad'])), 'reflect') for key, dset in datasets.items()}

    return datasets


def fetchfeeder(dataconf, givens=None):
    # Defaults
    givens = {} if givens is None else givens
    # Parse dataconf if required
    dataconf = path2dict(dataconf)

    # Build preptrain
    preptrains = buildpreptrains(dataconf['prepconfig'])

    # Load data to RAM (this is very doable, unlike FIB25)
    if ffd(givens, 'datasets') is not None:
        datasets = givens['datasets']
    else:
        datasets = load(dataconf['loadconfig'])

    # Make feeders (only membranes for now). The feeder structure is as follows:
    # X --- pX ---
    #             \                    X
    #              > --- pXY --- G --- | --- XYW --->
    #             /                    Y
    # Y --- pY ---

    # Build ground-truth feeder
    gt = ndk.cargo(data=datasets['gt'],
                   axistags='kij', nhoodsize=dataconf['nhoodsize'], stride=dataconf['stride'],
                   ds=dataconf['ds'], batchsize=dataconf['batchsize'], window=['x', 'x', 'x'],
                   preptrain=preptrains['pY'])

    # Build raw data feeder
    rd = gt.clonecrate(data=datasets['raw'], syncgenerators=True)
    rd.preptrain = preptrains['pX']

    # Zip feeders (weightmaps come from wmapmaker in preptrains['XY'])
    zippedfeeder = ndk.feederzip([rd, gt], preptrain=preptrains['pXY'])

    # Gate feeder
    if dataconf['prepconfig']['makewmap']:
        # Gate only if a weight map is available
        gate = tools.skipper
    else:
        gate = lambda inp: True

    gatepreptrain = pk.preptrain([pk.funczip((preptrains['X'], preptrains['Y'])), preptrains['XYW']])
    feeder = ndk.feedergate(zippedfeeder, gate, preptrain=gatepreptrain)

    # Weave feeders if required
    weave = givens['weave'] if 'weave' in givens.keys() else dataconf['weave'] if 'weave' in dataconf.keys() else False

    if weave:
        # Check if weavable
        assert (datasets['raw'].shape[0] == datasets['raw'].shape[1] == datasets['raw'].shape[2]), "Can only weave " \
                                                                                                   "cubical datasets."
        # Transpose datasets (201 and 120, assuming this orientation is 012)
        transposed = [{key: dset.transpose(transpose) for key, dset in datasets.items()}
                      for transpose in [(1, 2, 0), (2, 0, 1)]]
        # Get list of transposed feeders
        transposedfeeders = [feeder] + [fetchfeeder(dataconf, givens={'datasets': dataset, 'weave': False})
                                        for dataset in transposed]
        # Weave 'em all together and return
        weavedfeeder = ndk.feederweave(transposedfeeders)
        return weavedfeeder
    else:
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
        batches = feeder.next()
        batches = feeder.next()

        # Print
        for n, batch in enumerate(batches):
            print("Printing object {} of shape {} to file...".format(n, batch.shape))
            vz.printensor2file(batch, savedir=dataconf['plotdir'], mode='image', nameprefix='N{}--'.format(n))
            # np.save(dataconf['plotdir'] + 'N{}'.format(n), batch)

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
