import theano as th
import theano.tensor as T
import numpy as np

import Antipasti.netkit as nk
import Antipasti.netarchs as na
import Antipasti.archkit as ak
import Antipasti.netools as ntl
import Antipasti.netrain as nt


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
bn = lambda numinp=None: (nk.batchnormlayer(2, 0.9) if numinp is None else
                          nk.batchnormlayer(2, 0.9, inpshape=[None, numinp, None, None]))

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
def terminate(numout=3, finalactivation='softmax'):

    # Parse final layer
    if finalactivation == 'softmax' or finalactivation is None:
        fl = cll(4 * numout, numout, [1, 1]) + sml()
    elif finalactivation == 'linear':
        fl = cll(4 * numout, numout, [1, 1])
    elif finalactivation == 'sigmoid':
        fl = cls(4 * numout, numout, [1, 1])
    else:
        raise NotImplementedError

    # Fuse all losses
    fuse = trks(iusl((8, 8)), iusl((4, 4)), iusl((2, 2)), idl()) + merl(4) + fl

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
def initiate(preinit=None, numinp=None):

    if preinit is None:
        # Unpool and distribute
        start = repl(4) + trks(smpl((8, 8)), smpl((4, 4)), smpl((2, 2)), idl())
    elif preinit == 'batchnorm':
        # Apply bn, unpool and distribute
        start = bn(numinp=numinp) + repl(4) + trks(smpl((8, 8)), smpl((4, 4)), smpl((2, 2)), idl())
    else:
        raise NotImplementedError

    return start


# Build network from multiple blocks
def build(N=30, depth=5, transfer=None, parampath=None, numinp=3,
          numout=3, finalactivation='softmax', optimizer='momsgd', usewmap=True, savedir=None):

    print("[+] Building Cantor Network of depth {} and base width {} with {} inputs and {} outputs.".format(depth, N, numinp, numout))

    if transfer is not None:
        print("[+] Using transfer layer: {}.".format(transfer))

    if transfer == 'dropout':
        transfer = drl
    elif transfer == 'batchnorm':
        transfer = bn
    else:
        transfer = idl

    term = terminate

    print("[+] Activation of the final layer is set to: {}.".format(finalactivation))

    net = initiate(numinp=numinp) + \
          block(N=N, pos='start', numinp=numinp) + trks(transfer(), transfer(), transfer(), transfer()) + \
          reduce(lambda x, y: x + y, [block(N=N) +
                                      trks(transfer(), transfer(), transfer(), transfer())
                                      for _ in range(depth)]) + \
          block(N=N, pos='stop', numout=numout) + \
          term(numout=numout, finalactivation=finalactivation)

    net.feedforward()

    net = prep(net, parampath=parampath, usewmap=usewmap, savedir=savedir, optimizer=optimizer)

    return net


# Prepare network
def prep(net, parampath=None, optimizer='momsgd', usewmap=True, savedir=None):

    # Load params if required to
    if parampath is not None:
        net.load(parampath)
        print("[+] Loaded parameters from {}.".format(parampath))
    else:
        print("[-] Not loading parameters.")

    net.baggage["learningrate"] = th.shared(value=np.float32(0.0002))

    # Set up weight maps if required
    if usewmap:
        net.baggage["wmap"] = T.tensor4()
        net.cost(method='cce', wmap=net.baggage['wmap'], regterms=[(2, 0.0005)])
    else:
        net.cost(method='cce', regterms=[(2, 0.0005)])

    if optimizer == 'momsgd':
        net.getupdates(method=optimizer, learningrate=net.baggage["learningrate"], nesterov=True)
    elif optimizer == 'adam':
        net.getupdates(method=optimizer, learningrate=net.baggage["learningrate"])

    # Compute errors
    net = error(net)

    # Assign Save directory.
    if savedir is not None:
        net.savedir = savedir

    return net


# Compute theano error
def error(net):
    ist = net.y
    soll = net.yt
    # This flattens ist and soll to a matrix of shape (bs * nrow * ncol, nc)
    ist, soll = nt.prep(ist=ist, soll=soll)
    # Compute error
    E = T.neq(T.argmax(ist), T.argmax(soll)).mean()
    # Write to net
    net.E = E
    return net

