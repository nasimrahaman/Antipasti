__author__ = "nasim.rahaman@iwr.uni-heidelberg.de"
__doc__ = """General script for running networks."""


def pathsy(path):
    """Parse paths."""
    # This file is .../snemi/Scripts/train.py
    thisdirectory = os.path.dirname(__file__)
    # This is the SNEMI directory. path must be relative to this path
    snemihome = os.path.normpath(thisdirectory + '/../')
    # Target path
    outpath = os.path.join(snemihome, path)
    return outpath


def fetchfromdict(dictionary, key, default=None):
    """Try to fetch `dictionary[key]` if possible, return `default` otherwise."""
    if key in dictionary.keys():
        return dictionary[key]
    else:
        return default


def run(net, trX, **runconfig):
    """Given an Antipasti model object `net` and an Antipasti datafeeder `trX`, configure callbacks and fit model."""
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
        relay = tk.relay({'learningrate': net.baggage['learningrate']},
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

        def outputprinter(**iterstat):
            if iterstat['iternum'] % runconfig['live-print']['every'] == 0:
                # Get input and output
                ny = iterstat['y']
                bX = iterstat['funin'][0]
                bY = iterstat['funin'][1]
                # Print
                vz.printensor2file(ny, savedir=runconfig['live-print']['printdir'], mode='image',
                                   nameprefix='PR-'.format(iterstat['iternum']))
                vz.printensor2file(bX, savedir=runconfig['live-print']['printdir'], mode='image',
                                   nameprefix='RW-'.format(iterstat['iternum']))
                vz.printensor2file(bY, savedir=runconfig['live-print']['printdir'], mode='image',
                                   nameprefix='GT-'.format(iterstat['iternum']))
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

    # No weight maps
    extrarguments = {}

    # --- Breakpoint
    # import pdb
    # pdb.set_trace()

    # Set save directory
    net.savedir = runconfig.get('savedir')

    # Fit
    res = net.fit(trX=trX, numepochs=20, verbosity=5, backupparams=ffd(runconfig, 'backup-every', 200), log=log,
                  trainingcallbacks=cbs, extrarguments=extrarguments, extraoutputs=extraoutputs, relay=relay)

    if ffd(runconfig, 'picklejar') is not None:
        nu.pickle(res, os.path.join(runconfig['picklejar'], 'fitlog.save'))

    return net


def plot(net, trX, **plotconfig):
    """Plot intermediate results given a model `net` and a datafeeder `trX`."""

    # Glob params for smoother UI
    plotconfig['params'] = glob.glob(plotconfig['params'])[0]
    # plotconfig['params'] could be a directory. If that's the case, select the most recent parameter file and load
    if os.path.isdir(plotconfig['params']):
        print("[-] Given parameter file is a directory. Will fetch the most recent set of parameters.")
        # It's a dir
        # Get file name of the most recent file
        ltp = sorted(os.listdir(plotconfig['params']))[-1]
        parampath = os.path.join(plotconfig['params'], ltp)
    else:
        # It's a file
        parampath = plotconfig['params']
        pass

    print("[+] Loading parameters from {}.".format(parampath))

    # Load params
    net.load(parampath)

    # Get batches from feeders
    batches = [trX.next() for _ in range(plotconfig['numbatches'])]

    for n, batch in enumerate(batches):
        print("[+] Evaluating batch {}...".format(n))
        bX, bY = batch[0:2]

        ny = net.y.eval({net.x: bX})
        vz.printensor2file(bX, savedir=plotconfig['plotdir'], mode='image', nameprefix='RD{}--'.format(n))
        vz.printensor2file(bY, savedir=plotconfig['plotdir'], mode='image', nameprefix='GT{}--'.format(n))
        vz.printensor2file(ny, savedir=plotconfig['plotdir'], mode='image', nameprefix='PR{}--'.format(n))

    print("[+] Plotted images to {}.".format(plotconfig['plotdir']))

    print("[+] Done.")


if __name__ == '__main__':
    print("[+] Initializing...")
    import argparse
    import yaml
    import sys
    import os
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

    print("[+] Using configuration file from {}.".format(args.configset))

    # Read which device to use
    if args.device is None:
        device = config['device']
    else:
        device = args.device

    assert device is not None, "Please provide the device to be used as a bash argument " \
                               "(e.g.: python train.py /path/to/config/file.yml --device gpu0)"

    print("[+] Using device {}.".format(device))

    # Import shit
    from theano.sandbox.cuda import use
    use(device)

    # Add Antipasti to path
    sys.path.append('/export/home/nrahaman/Python/Repositories/Antipasti')
    import Antipasti.trainkit as tk
    import Antipasti.netutils as nu
    import Antipasti.vizkit as vz

    print("[+] Importing model and datafeeders...")
    # Import model
    model = imp.load_source('model', config['modelpath'])
    # Import datafeeder
    dpl = imp.load_source('dataplate', config['dplpath'])

    print("[+] Building network...")
    # Build network
    net = model.build(**config['buildconfig'])

    print("[+] Fetching feeders with configuration at {}.".format(config['dataconf']))
    # Fetch datafeeders
    trX = dpl.fetchfeeder(config['dataconf'])

    if 'runconfig' in config.keys():
        print("[+] Ready to run.")
        # Run training
        run(net, trX, **config['runconfig'])
    elif 'plotconfig' in config.keys():
        print("[+] Ready to plot.")
        # Plot
        plot(net, trX, **config['plotconfig'])
