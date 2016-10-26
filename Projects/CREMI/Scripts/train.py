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
        print("[+] Logging to {}.".format(runconfig['logfile']))
    else:
        log = None
        print("[-] Not logging.")

    # Configure relay
    if ffd(runconfig, 'relayfile') is not None:
        relay = tk.relay({'learningrate': net.baggage['learningrate'],
                          'l2': net.baggage['l2'],
                          'vgg-learningrate': net.baggage['vgg-learningrate']},
                         runconfig['relayfile'])
        print("[+] Listening to control file at {}.".format(runconfig['relayfile']))
    else:
        relay = None
        print("[-] Not listening to relays.")

    # Build a list of callbacks (start with printer)
    cbl = [tk.makeprinter(verbosity=5)]

    # Print every few iterations
    if ffd(runconfig, 'live-print') is not None:

        print("[+] Network outputs will be printed "
              "to {} every {} iteratons.".format(runconfig['live-print']['printdir'], runconfig['live-print']['every']))

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
        print("[+] Live-plots are on. Make sure a bokeh-server is running.")
        cbl.append(tk.plotter(linenames=runconfig['live-plot']['linenames'], colors=runconfig['live-plot']['colors']))
    else:
        pass

    # Build callbacks
    # Training
    cbs = tk.callbacks(cbl)
    # Validation
    vcbs = tk.printer(monitors=[tk.vaEmonitor], textlogger=log)

    # Bind textlogger to printer
    cbs.callbacklist[0].textlogger = log

    if runconfig['use-wmaps']:
        extrarguments = {net.baggage['wmap']: -1}
    else:
        extrarguments = {}

    # Fit
    if isinstance(trX, dict):
        trX_, vaX_ = trX['training'], trX['validation']
        res = net.fit(trX=trX_, numepochs=40, verbosity=5, backupparams=ffd(runconfig, 'backup-every', 200), log=log,
                      trainingcallbacks=cbs, extrarguments=extrarguments, extraoutputs=extraoutputs, relay=relay,
                      vaX=vaX_, validateevery=ffd(runconfig, 'validate-every', 200), validationcallbacks=vcbs)
    else:
        res = net.fit(trX=trX, numepochs=40, verbosity=5, backupparams=ffd(runconfig, 'backup-every', 200), log=log,
                      trainingcallbacks=cbs, extrarguments=extrarguments, extraoutputs=extraoutputs, relay=relay)

    if ffd(runconfig, 'picklejar') is not None:
        nu.pickle(res, os.path.join(runconfig['picklejar'], 'fitlog.save'))

    return net


def plot(net, trX, **plotconfig):
    # Load params
    parampath = glob.glob(plotconfig['parampath'])[0]

    print("[+] Loading parameters from {}.".format(parampath))
    net.load(parampath)
    print("[+] Printing to {}.".format(plotconfig['plotdir']))

    # Get validation feeder from trX
    if isinstance(trX, dict):
        trX = trX[plotconfig.get('dataset', 'validation')]

    # Get batches to plot
    batches = [trX.next() for _ in range(plotconfig['numbatches'])]
    # Infer on the batches
    for batchnum, batch in enumerate(batches):
        bn = batchnum if len(batches) > 1 else ''
        print("[+] Inferring on batch {}.".format(batchnum))
        # Fetch and infer
        batchX, batchY = batch[0:2]
        # Infer
        ny = net.y.eval({net.x: batchX})
        # Print
        vz.printensor2file(ny, savedir=plotconfig['plotdir'], mode='image', nameprefix='PR-{}-'.format(bn))
        vz.printensor2file(batchX, savedir=plotconfig['plotdir'], mode='image', nameprefix='RW-{}-'.format(bn))
        vz.printensor2file(batchY, savedir=plotconfig['plotdir'], mode='image', nameprefix='GT-{}-'.format(bn))
    print("[+] Done.")


if __name__ == '__main__':
    import argparse
    import yaml
    import sys
    import imp
    import os
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
        if config.get('mode', 'run') == 'run':
            run(net, trX, **config['runconfig'])
        elif config.get('mode') == 'plot':
            plot(net, trX, **config['plotconfig'])
        else:
            raise NotImplementedError
    except Exception as e:
        raise e
    finally:
        if not isinstance(trX, dict):
            trX.cleanup()
        else:
            for _, feeder in trX.items():
                feeder.cleanup()