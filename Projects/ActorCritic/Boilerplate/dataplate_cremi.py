# Add to path
import sys

sys.path.append('/export/home/nrahaman/Python/Antipasti/Projects/ActorCritic/Boilerplate')
import prepfunctions_cremi

import Antipasti.trainkit as tk
import Antipasti.netdatautils as ndu
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
    pf = prepfunctions_cremi.prepfunctions()

    # Build preptrain for raw data
    ptX = pk.preptrain([pk.im2double(nbit=8), pk.cast(), pk.normalizebatch(), pf['time2channel']])

    # Build preptrain for ground truth
    ptY = pk.preptrain([pf['time2channel'], pf['seg2membrane'](), pf['disttransform'](gain=prepconfig['edt']),
                        pk.cast()])

    # Build preptrain for the zipped XY feeder
    ptXY = pk.preptrain([])

    # Elastic transformations if requested
    if prepconfig['elastic']:
        ptXY.append(pf['elastictransform'](**prepconfig['elastic-params']))

    # Random rotate if requested
    if prepconfig['random-rotate']:
        ptXY.append(pf['randomrotate']())

    # Random flip if requested
    if prepconfig['random-flip']:
        ptXY.append(pf['randomflip']())

    if prepconfig['random-flip-z']:
        ptXY.append(pf['randomflipz']())

    return {'X': ptX, 'Y': ptY, 'XY': ptXY}


def load(loadconfig):
    loadconfig = path2dict(loadconfig)
    # Load from H5
    datasets = {dsetname: {dsetobj: ndu.fromh5(path=dsetname,
                                               datapath=loadconfig['h5paths'][dsetname][dsetobj],
                                               dataslice=loadconfig['slices'][dsetname])
                           for dsetobj in ['raw', 'gt', 'syn'] if loadconfig['h5paths'][dsetname][dsetobj] is not 'x'}
                for dsetname, dsetpath in loadconfig['paths'].items()}

    return datasets


def fetchfeeder(dataconf):
    dataconf = path2dict(dataconf)
    # Load datasets
    datasets = load(dataconf['loadconfig'])
    # TODO
    pass


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
