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

    # Build a list of callbacks (start with printer)
    cbl = [tk.makeprinter(verbosity=5)]

    # Print every few iterations
    if ffd(runconfig, 'live-print') is not None:
        # the network output must be available
        extraoutputs = {'y': net.y}

        def outputprinter(**kwargs):
            if kwargs['iternum'] % runconfig['live-print']['every'] == 0:
                # Get input and output
                ny = kwargs['y']
                bX = kwargs['funin'][0]
                bY = kwargs['funin'][1]
                # Print
                vz.printensor2file(ny, savedir=runconfig['live-print']['printdir'], mode='image',
                                   nameprefix='PR-'.format(kwargs['iternum']))
                vz.printensor2file(bX, savedir=runconfig['live-print']['printdir'], mode='image',
                                   nameprefix='RW-'.format(kwargs['iternum']))
                vz.printensor2file(bY, savedir=runconfig['live-print']['printdir'], mode='image',
                                   nameprefix='GT-'.format(kwargs['iternum']))
            else:
                return

        # Append to callback list
        cbl.append(tk.caller(outputprinter))

    else:
        extraoutputs = {}

    # Live plots
    if ffd(runconfig, 'live-plot') is not None:
        cbl.append(tk.plotter(linenames=runconfig['live-plot']['linenames'], colors=runconfig['live-plot']['colors']))
    else:
        pass

    # Build callbacks
    cbs = tk.callbacks(cbl)

    # Bind textlogger to printer
    cbs.callbacklist[0].textlogger = log

    if runconfig['use-wmaps']:
        extrarguments = {net.baggage['wmap']: -1}
    else:
        extrarguments = {}

    # Fit
    res = net.fit(trX=trX, numepochs=40, verbosity=5, backupparams=200, log=log, trainingcallbacks=cbs,
                  extrarguments=extrarguments, extraoutputs=extraoutputs, relay=relay)

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
        run(net, trX, **config['runconfig'])
    except Exception as e:
        raise e
    finally:
        trX.cleanup()
