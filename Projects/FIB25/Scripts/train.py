__doc__ = """General script for running networks."""



def fetchfromdict(dictionary, key, default=None):
    if key in dictionary.keys():
        return dictionary[key]
    else:
        return default


def run(net, trX, **runconfig):
    # Aliases
    ffd = fetchfromdict

    # Configure logfile
    if ffd(runconfig, 'logfile') is not None:
        log = tk.logger(runconfig['logfile'])
    else:
        log = None

    # Configure relay
    if ffd(runconfig, 'relayfile') is not None:
        relay = tk.relay({'learningrate': net.baggage['learningrate']}, runconfig['relayfile'])
    else:
        relay = None

    # Build callbacks
    cbs = tk.callbacks([tk.makeprinter(verbosity=5), tk.plotter(linenames=['C', 'L'], colors=['navy', 'firebrick'])])

    # Bind textlogger to printer
    cbs.callbacklist[0].textlogger = log

    # Fit
    res = net.fit(trX=trX, numepochs=40, verbosity=5, backupparams=200, log=log, trainingcallbacks=cbs,
                  extrarguments={}, relay=relay)

    if ffd(runconfig, 'picklejar') is not None:
        nu.pickle(res, runconfig['picklejar'] + 'fitlog.save')

    return net


if __name__ == '__main__':
    import argparse
    import yaml
    import sys
    import imp

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

    # Import model
    model = imp.load_source('model', config['modelpath'])
    # Import datafeeder
    dpl = imp.load_source('dataplate', config['dplpath'])

    # Build network
    net = model.build(**config['buildconfig'])

    # Fetch datafeeders
    trX = dpl.fetchfeeder(config['dataconf'])

    # Run training
    run(net, trX, **config['runconfig'])
