__doc__ = """General script for running networks."""



def fetchfromdict(dictionary, key, default=None):
    if key in dictionary.keys():
        return dictionary[key]
    else:
        return default


def plot(net, trX, **plotconfig):
    print("[+] Plotting results in {}.".format(plotconfig['plotdir']))

    # Get parameter file (first element after a glob) and load parameters
    parampath = glob.glob(plotconfig['parampath'])[0]
    print("[+] Loading parameters from {}".format(parampath))

    net.load(parampath)

    print("[+] Fetching from feeder...")
    # Fetch batches
    batches = [trX.next() for _ in range(plotconfig['numbatches'])]

    # Loop over and evaluate
    for n, batch in enumerate(batches):
        # Read from batch
        bX, bY, bW = batch

        print("[+] Evaluating batch {} of {}...".format(n, plotconfig['numbatches']))
        # Evaluate
        ny = net.y.eval({net.x: bX.astype('float32')})
        # Plot
        vz.printensor2file(bX, savedir=plotconfig['plotdir'], mode='image', nameprefix='RAW{}--'.format(n))
        vz.printensor2file(bY, savedir=plotconfig['plotdir'], mode='image', nameprefix='GT{}--'.format(n))
        vz.printensor2file(ny, savedir=plotconfig['plotdir'], mode='image', nameprefix='PRED{}--'.format(n))
        vz.printensor2file(bW, savedir=plotconfig['plotdir'], mode='image', nameprefix='WM{}--'.format(n))

    # Done
    print("[+] Done.")

if __name__ == '__main__':
    import argparse
    import yaml
    import sys
    import imp
    import glob

    # Parse arguments
    parsey = argparse.ArgumentParser()
    parsey.add_argument("configset", help="Configuration file.")
    parsey.add_argument("--device", help="Device to use (overrides configuration file).", default=None)
    args = parsey.parse_args()

    # Load configuration dict
    with open(args.configset) as configfile:
        config = yaml.load(configfile)

    # Read which device to use
    if args.device is None:
        device = config['device']
    else:
        device = args.device

    assert device is not None

    # Import shit
    from theano.sandbox.cuda import use
    use(device)

    # Add Antipasti to path
    sys.path.append('/export/home/nrahaman/Python/Antipasti')
    import Antipasti.trainkit as tk
    import Antipasti.netutils as nu
    import Antipasti.vizkit as vz

    # Import model
    model = imp.load_source('model', config['modelpath'])
    # Import datafeeder
    dpl = imp.load_source('dataplate', config['dplpath'])

    # Build network
    net = model.build(**config['buildconfig'])

    # Fetch datafeeders
    trX = dpl.fetchfeeder(config['dataconf'])

    # Run training
    try:
        plot(net, trX, **config['plotconfig'])
    except Exception as e:
        raise e
    finally:
        trX.cleanup()
