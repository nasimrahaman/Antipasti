import argparse
import os
import yaml
import socket

DEFAULT_CONFIG_PATH = {"littleheron": "/mnt/localdata02/nrahaman/Neuro/ConvNet-Backups/CREMI/SIERPINSKI-SYNAPSES/"
                                      "Configurations/runconfigset.yml",
                       "fatchicken": "/home/nrahaman/LittleHeronHDD2/Neuro/ConvNet-Backups/CREMI/SIERPINSKI-SYNAPSES/"
                                     "Configurations/runconfigset.yml"}

# Parse configuration set
parser = argparse.ArgumentParser(description="Training Script for Cantor (SIERPINSKI) Networks")
parser.add_argument('--configset', help='Possible keys are: [A], [B], [DROPSLICE]',
                    default=DEFAULT_CONFIG_PATH[socket.gethostname()])
args = parser.parse_args()
configset = args.configset

# Set up Configurations
if os.path.exists(configset):
    with open(configset, 'r') as f:
        config = yaml.load(f)
else:
    raise NotImplementedError("Unknown configset: {}".format(configset))



from theano.sandbox.cuda import use
use(config['device'])
import theano as th
import numpy as np

import Antipasti.netkit as nk
import Antipasti.netarchs as na
import Antipasti.archkit as ak
import Antipasti.netools as ntl
import Antipasti.netutils as nu
import Antipasti.trainkit as tk
import Antipasti.vizkit as vz

import imp

print("[+] Configuration Set is '{}'.".format(configset))

if 'info' in config.keys():
    print("[+] Configuration info-tag says: {}".format(config['info']))
else:
    print("[-] No configuration info-tag found.")

# Define shortcuts
# Convlayer with ELU
cl = lambda fmapsin, fmapsout, kersize: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout, kersize=kersize,
                                                     activation=ntl.elu())

# Convlayer without activation
cll = lambda fmapsin, fmapsout, kersize: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout, kersize=kersize)

# Convlayer with Sigmoid
cls = lambda fmapsin, fmapsout, kersize: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout, kersize=kersize,
                                                      activation=ntl.sigmoid())

# Strided convlayer with ELU (with autopad)
scl = lambda fmapsin, fmapsout, kersize, padding=None: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout,
                                                                    kersize=kersize,
                                                                    stride=[2, 2], activation=ntl.elu(),
                                                                    padding=padding)

# Strided 3x3 pool layerlayertrain or Antipasti.netarchs.layertrainyard
spl = lambda: nk.poollayer(ds=[3, 3], stride=[2, 2], padding=[1, 1])

# Strided 3x3 mean pool layer
smpl = lambda ds=(2, 2): nk.poollayer(ds=list(ds), poolmode='mean')

# 2x2 Upscale layer
usl = lambda us=(2, 2): nk.upsamplelayer(us=list(us))

# 2x2 Upscale layer with interpolation
iusl = lambda us=(2, 2): nk.upsamplelayer(us=list(us), interpolate=True)

# Batch-norm layer
bn = lambda: nk.batchnormlayer(2, 0.9)

# Softmax
sml = lambda: nk.softmax(dim=2)

# Identity
idl = lambda: ak.idlayer()

# Replicate
repl = lambda numrep: ak.replicatelayer(numrep)

# Merge
merl = lambda numbranch: ak.mergelayer(numbranch)

# Split in half
sptl = lambda splitloc: ak.splitlayer(splits=splitloc, dim=2, issequence=False)

# Dropout layer
drl = lambda p=0.5: nk.noiselayer(noisetype='binomial', p=p)

# Addition layer
adl = lambda numinp: ak.addlayer(numinp, dim=2, issequence=False)

# Circuit layer
crcl = lambda circuit: ak.circuitlayer(circuit, dim=2, issequence=False)

# Parallel tracks
trks = lambda *layers: na.layertrainyard([list(layers)])

lty = lambda ty: na.layertrainyard(ty)


# Polarity = 'e' or 'c' for 'expand' or 'contract'

def to2prong(layer, mode, primary):
    # mode in ['u' or 'd']
    assert mode in ['u', 'd']
    newlayer = layer + repl(2) + (idl() * primary() if mode == 'd' else primary() * idl())
    return newlayer

t2p = to2prong
pd = t2pd = lambda layer: to2prong(layer, 'd', iusl)
pu = t2pu = lambda layer: to2prong(layer, 'u', spl)

def to3prong(layer, primary=None, secondary=None):
    if primary is None:
        primary = spl
    if secondary is None:
        secondary = iusl
    newlayer = layer + repl(3) + na.layertrainyard([[primary(), idl(), secondary()]])
    return newlayer

t3p = to3prong

# Define Layer Types

def block(N=50, pos='mid', numinp=3, numout=3, legacy=False):
    # Define layer types
    lA0 = lambda: cl(8 * N, 8 * N, [3, 3])
    lB0 = lambda: cl(12 * N, 8 * N, [3, 3])

    lA1 = lambda: cl(12 * N, 4 * N, [5, 5])
    lB1 = lambda: cl(14 * N, 4 * N, [5, 5])

    lA2 = lambda: cl(6 * N, 2 * N, [7, 7])
    lB2 = lambda: cl(7 * N, 2 * N, [7, 7])

    lA3 = lambda: cl(3 * N, 1 * N, [9, 9])

    # Define initiator layers
    lA0i = lambda: cl(numinp, 8 * N, [3, 3])
    lA1i = lambda: cl(numinp, 4 * N, [5, 5])
    lA2i = lambda: cl(numinp, 2 * N, [7, 7])
    lA3i = lambda: cl(numinp, 1 * N, [9, 9])

    # Legacy: the time before numout was introduced
    if legacy:
        numout = 3

    # Define terminator layers
    lA0t = lambda: cl(8 * N, numout, [3, 3])
    lA1t = lambda: cl(12 * N, numout, [5, 5])
    lA2t = lambda: cl(6 * N, numout, [7, 7])
    lA3t = lambda: cl(3 * N, numout, [9, 9])

    # Define circuits
    cA = lambda: crcl([[0, 1], [2, 3], [4, 5], 6])
    cB = lambda: crcl([0, [1, 2], 3, 4])
    cC = lambda: crcl([[0, 1], 2, [3, 4], 5])
    cD = lambda: crcl([[0, 1], [2, 3], 4, [5, 6]])

    blk = ((trks(lA0i(), pu(lA1i()), pu(lA2i()), pu(lA3i())) + cA()) if pos == 'start'
           else trks(lA0(), pu(lA1()), pu(lA2()), pu(lA3())) + cA()) + \
           trks(pd(lB0()), idl(), idl(), idl()) + cB() + \
           trks(lA0(), t3p(lB1()), idl(), idl()) + cC() + \
           trks(pd(lB0()), idl(), idl(), idl()) + cB() + \
           trks(lA0(), pu(lA1()), t3p(lB2()), idl()) + cD() + \
           trks(pd(lB0()), idl(), idl(), idl()) + cB() + \
           trks(lA0(), t3p(lB1()), idl(), idl()) + cC() + \
           ((trks(pd(lB0()), idl(), idl(), idl()) + cB() + trks(lA0t(), lA1t(), lA2t(), lA3t())) if pos == 'stop'
            else (trks(pd(lB0()), idl(), idl(), idl()) + cB()))

    return blk

# Terminate a network
def terminate(numout=3):
    # Fuse all losses
    fuse = trks(iusl((8, 8)), iusl((4, 4)), iusl((2, 2)), idl()) + merl(4) + cls(4 * numout, numout, [1, 1])
    return fuse

# The old, deprecated way of terminating the network. There because that's how a few networks were
# trained.
def legacyterminate(numout=3):
    # Fuse all losses
    fuse = trks(iusl((8, 8)), iusl((4, 4)), iusl((2, 2)), idl()) + merl(4) + cls(12, numout, [1, 1])
    return fuse

# Fuse Terminate
def fuseterminate(numout=3):
    # Fuse
    fuse = trks(iusl((4, 4)) + cl(numout, numout, [9, 9]) + iusl((2, 2)) + cl(numout, numout, [5, 5]),
                iusl((4, 4)) + cl(numout, numout, [9, 9]),
                iusl((2, 2)) + cl(numout, numout, [5, 5]),
                cl(numout, numout, [5, 5])) + \
           adl(4) + \
           cl(numout, numout, [3, 3]) + cls(numout, numout, [3, 3])

    return fuse

# Initiate a network
def initiate():
    # Unpool and distribute
    start = repl(4) + trks(smpl((8, 8)), smpl((4, 4)), smpl((2, 2)), idl())
    return start

# Build network from multiple blocks
def build(N=50, depth=2, transfer=None, fuseterm=False, parampath=None, iterstart=0, numinp=3,
          numout=3, legacy=False):

    print("[+] Building Cantor Network of depth {} and base width {} with {} inputs and {} outputs.".format(depth, N, numinp, numout))

    if transfer is not None:
        print("[+] Using transfer protocol: {}.".format(transfer))

    if transfer == 'dropout':
        transfer = drl
    elif transfer == 'batchnorm':
        transfer = bn
    else:
        transfer = idl

    if fuseterm:
        term = fuseterminate
        print("[+] Using fusion termination.")
    elif legacy:
        term = legacyterminate
        print("[-] Using legacy termination.")
    else:
        term = terminate
        print("[+] Using vanilla termination.")

    net = initiate() + block(N=N, pos='start', numinp=numinp) + \
          reduce(lambda x, y: x + y, [trks(transfer(), transfer(), transfer(), transfer()) +
                                      block(N=N) +
                                      trks(transfer(), transfer(), transfer(), transfer())
                                      for _ in range(depth)]) + \
          block(N=N, pos='stop', numout=numout, legacy=legacy) + term(numout=numout)

    net.feedforward()

    if parampath is not None:
        net.load(parampath)

    net.cost(method='bce', regterms=[(2, 0.0005)])

    # Shared variable for learningrate
    learningrate = th.shared(np.float32(0.0002))
    # Add to baggage
    net.baggage['learningrate'] = learningrate
    # Get updates
    net.getupdates(method='adam', learningrate=learningrate)

    return net

def build_ICV1():
    # Kill dropout
    drl = idl

    # Inception module
    def inceptionize(streams):
        # Compute number of streams
        numstreams = len(streams)
        # Multiply
        module = na.layertrainyard([streams])
        # Build replicate and merge layers
        rep = ak.replicatelayer(numstreams)
        mer = ak.mergelayer(numstreams)
        # Build and return inception module
        return rep + module + mer

    # Build the networkdparam), 0., dparam) for dparam in dC]
    # Return
    # --- a1 --- b1 --- --- c1 --- d1 --- d2 --- c2 --- --- b1 --- a1 ---
    #                  |                               |
    #                   ------------- id --------------

    a1 = cl(3, 32, [9, 9]) + drl() + cl(32, 48, [9, 9])

    b1 = scl(48, 128, [7, 7]) + drl() + \
         inceptionize([cl(128, 64, [3, 3]) + cl(64, 64, [1, 1]), cl(128, 64, [5, 5]) + cl(64, 64, [3, 3])]) + \
         cl(128, 160, [3, 3])

    c1 = inceptionize([cl(160, 64, [5, 5]) + spl(), scl(160, 64, [3, 3]) + cl(64, 96, [1, 1])]) + \
         cl(160, 160, [3, 3]) + drl() + \
         inceptionize([cl(160, 100, [7, 7]), cl(160, 48, [5, 5]) + cl(48, 48, [1, 1]),
                       cl(160, 64, [3, 3]) + cl(64, 64, [1, 1])]) + \
         cl(212, 240, [3, 3])

    d1 = inceptionize([cl(240, 192, [1, 1]) + spl(), scl(240, 512, [3, 3])]) + cl(704, 1024, [3, 3])

    d2 = drl() + inceptionize([cl(1024, 384, [3, 3]) + cl(384, 200, [3, 3]), cl(1024, 260, [1, 1]),
                               cl(1024, 384, [5, 5]) + cl(384, 200, [1, 1])]) + \
         cl(660, 512, [3, 3]) + \
         inceptionize([cl(512, 60, [7, 7]), cl(512, 180, [3, 3])]) + \
         usl()

    c2 = drl() + cl(240, 200, [3, 3]) + \
         inceptionize([cl(200, 140, [3, 3]) + cl(140, 80, [3, 3]), cl(200, 140, [5, 5]) + cl(140, 80, [5, 5])]) + \
         cl(160, 160, [5, 5]) + \
         usl()

    b2 = drl() + cl(320, 128, [5, 5]) + \
         inceptionize([cl(128, 60, [9, 9]) + cl(60, 48, [5, 5]), cl(128, 72, [5, 5]) + cl(72, 48, [5, 5])]) + \
         cl(96, 60, [5, 5]) + \
         cl(60, 48, [3, 3]) + \
         usl()

    a2 = drl() + cl(48, 32, [9, 9]) + cl(32, 16, [5, 5]) + cl(16, 16, [3, 3]) + cls(16, 3, [1, 1])

    # Putting it together
    interceptorv1 = a1 + b1 + repl(2) + (c1 + d1 + d2 + c2) * idl() + merl(2) + b2 + a2
    interceptorv1.feedforward()

    interceptorv1.cost(method='bce', regterms=[(2, 0.0005)])
    interceptorv1.getupdates(method='momsgd', learningrate=0.000005, nesterov=True)

    return interceptorv1

def run(net, dataconfig=None, logfile=None, weave=False, savedir=None, picklejar=None, relayfile=None):
    # Load data
    dplpaths = {'fatchicken': None,
                'littleheron': '/export/home/nrahaman/Python/Antipasti/Projects/FIB25/Boilerplate/dataplate.py'}

    if savedir is None:
        savedirs = {'fatchicken': None,
                    'littleheron': '/mnt/localdata02/nrahaman/Neuro/ConvNet-Backups/FIB25/SIERPINSKI/Weights'}
        savedir = savedirs[socket.gethostname()]

    if picklejar is None:
        picklejars = {'fatchicken': None,
                      'littleheron': '/mnt/localdata02/nrahaman/Neuro/ConvNet-Backups/FIB25/SIERPINSKI/PickleJar/'}
        picklejar = picklejars[socket.gethostname()]

    dpl = imp.load_source('dataplate', dplpaths[socket.gethostname()])

    if dataconfig is not None:
        print("[+] Using data configuration from {}".format(dataconfig))
    else:
        print("[-] Feeders will be called without a dataconfig.")

    if not weave:
        print("[+] Not using weaved feeders.")
        trX = dpl.fetchfeeder(dataconfig)
    else:
        print("[+] Using weaved feeders.")
        trX = dpl.fetchweavedfeeders(dataconfig)

    print("[+] Saving parameters to {}".format(savedir))
    # Save directory
    net.savedir = savedir

    if logfile is not None:
        print("[+] Logging progress to: {}".format(logfile))
        log = tk.logger(logfile)
    else:
        print("[-] Not logging progress.")
        log = None

    if relayfile is not None:
        assert 'learningrate' in net.baggage.keys()
        print("[+] Using relay file from: {}".format(relayfile))
        relay = tk.relay(switches={'learningrate': net.baggage['learningrate']}, ymlfile=relayfile)
    else:
        print("[-] No relay file given.")
        relay = None

    # Build callbacks
    cbs = tk.callbacks([tk.makeprinter(verbosity=5), tk.plotter(linenames=['C', 'L'], sessionname='CostLoss',
                                                                colors=['navy', 'firebrick'])])
    # Bind textlogger to printer
    cbs.callbacklist[0].textlogger = log

    # Fit
    res = net.fit(trX=trX, numepochs=40, verbosity=5, backupparams=200, log=log, trainingcallbacks=cbs, relay=relay)
    nu.pickle(res, picklejar + "fitlog.save")

def plot(net, dataconfig, plotdir=None):
    print("[+] Plotting network outputs.")
    # Load data
    dplpaths = {'fatchicken': None,
                'littleheron': '/export/home/nrahaman/Python/Antipasti/Projects/FIB25/Boilerplate/dataplate.py'}

    dpl = imp.load_source('dataplate', dplpaths[socket.gethostname()])

    print("[+] Fetching data with config from {}.".format(dataconfig))
    trX = dpl.fetchfeeder(dataconfig)

    if plotdir is None:
        plotdirs = {'littleheron': '/mnt/localdata02/nrahaman/Neuro/ConvNet-Backups/FIB25/SIERPINSKI/Plots/'}
        plotdir = plotdirs[socket.gethostname()]
        print("[-] Plot directory is not given. Using default.")

    print("[+] Saving images to {}.".format(plotdir))

    print("[+] Working on the first batch...")
    # Fetch next element from datafeeder
    batchX, batchY = trX.next()
    # Evaluate
    ny0 = net.y.eval({net.x: batchX})
    # Print
    vz.printensor2file(ny0, savedir=plotdir, mode='image', nameprefix='PRED0--')
    vz.printensor2file(batchX, savedir=plotdir, mode='image', nameprefix='RD0--')
    vz.printensor2file(batchY, savedir=plotdir, mode='image', nameprefix='GT0--')

    print("[+] Working on the second batch...")
    # Fetch next element from datafeeder
    batchX, batchY = trX.next()
    # Evaluate
    ny1 = net.y.eval({net.x: batchX})
    # Print
    vz.printensor2file(ny1, savedir=plotdir, mode='image', nameprefix='PRED1--')
    vz.printensor2file(batchX, savedir=plotdir, mode='image', nameprefix='RD1--')
    vz.printensor2file(batchY, savedir=plotdir, mode='image', nameprefix='GT1--')
    print("[+] Done.")


if __name__ == '__main__':
    hostname = socket.gethostname()

    # Parse from config
    mode = config['mode']

    # Parse based on mode
    if mode == 'run':
        loadparams = config['loadparams']
        iterstart = config['iterstart']
        # Host dependent paths
        parampath = config['parampath'][hostname] if loadparams else None
        logfile = config['logfile'][hostname]
        relayfile = config['relayfile'][hostname] if 'relayfile' in config.keys() else None
        dataconfig = config['dataconfig'][hostname]
        savedir = config['savedirs'][hostname]
        picklejar = config['picklejars'][hostname]
    elif mode == 'plot':
        loadparams = config['loadparams']
        plotdir = config['plotdir'][hostname]
        parampath = config['parampath'][hostname] if loadparams else None
        dataconfig = config['dataconfig'][hostname]
        iterstart = 0

    # Network build parameters
    buildparams = config['buildparams']

    # Run!
    if hostname == 'littleheron':
        if parampath is not None:
            print("[+] Using parameters from {}.".format(parampath))
        else:
            print("[-] Not using pretrained parameters.")

        print("[+] Building Model...")
        network = build(parampath=parampath, iterstart=iterstart, **buildparams)
        if mode == 'run':
            run(network, dataconfig=dataconfig, logfile=logfile, savedir=savedir, picklejar=picklejar,
                relayfile=relayfile)
        elif mode == 'plot':
            print("Plotting with parameters from {}...".format(parampath))
            plot(network, dataconfig, plotdir)
