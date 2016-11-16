
import theano as th
import theano.tensor as T
import numpy as np

import Antipasti.netkit as nk
import Antipasti.netarchs as na
import Antipasti.archkit as ak
import Antipasti.netools as ntl
import Antipasti.netrain as nt
import Antipasti.netutils as nu

import lasagne.layers as ll
import lasagne.nonlinearities as nl

# ---- Lasagne
# Convlayer with ELU
lcl = lambda incoming, numout, kersize: ll.Conv2DLayer(incoming, num_filters=numout, filter_size=kersize,
                                                       nonlinearity=nl.elu, pad='same')

# Convlayer without activation
lcll = lambda incoming, numout, kersize: ll.Conv2DLayer(incoming, num_filters=numout, filter_size=kersize,
                                                        nonlinearity=nl.linear, pad='same')

lcls = lambda incoming, numout, kersize: ll.Conv2DLayer(incoming, num_filters=numout, filter_size=kersize,
                                                        nonlinearity=nl.sigmoid, pad='same')

# Dilated convolution layer with ELU and 'same' border mode (but odd kernel size)
lcld = lambda incoming, numout, kersize, dilation: \
    ll.PadLayer(ll.DilatedConv2DLayer(incoming, num_filters=numout, filter_size=kersize, dilation=dilation,
                                      nonlinearity=nl.elu),
                width=(((kersize - 1)/2) * dilation))

# Strided pool layer
lspl = lambda incoming: ll.Pool2DLayer(incoming, pool_size=3, stride=2, pad=(1, 1))

# Upsample Layer
lusl = lambda incoming: ll.Upscale2DLayer(incoming, scale_factor=2)

# Add layer
ladl = lambda *incomings: ll.ElemwiseSumLayer(incomings)

# Concatenate layer
lmerl = lambda *incomings: ll.ConcatLayer(incomings)

# Input layer
linp = lambda fmapsin: ll.InputLayer((None, fmapsin, None, None))

# Identity layer (with args and kwargs)
lidl = lambda incoming, *args, **kwargs: ll.NonlinearityLayer(incoming, nonlinearity=nl.identity)

# ---- Antipasti

# Convlayer with ELU
cl = lambda fmapsin, fmapsout, kersize: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout, kersize=kersize,
                                                     activation=ntl.elu())

clv = lambda fmapsin, fmapsout, kersize: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout, kersize=kersize,
                                                      convmode='same', activation=ntl.elu())

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

# Strided convlayer without nonlinearity (with autopad)
scll = lambda fmapsin, fmapsout, kersize, padding=None: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout,
                                                                    kersize=kersize,
                                                                    stride=[2, 2], activation=ntl.linear(),
                                                                    padding=padding)

# Activation layer
acl = lambda fn=ntl.elu(): ak.activationlayer(fn)

# Identity
idl = lambda: ak.idlayer()

# Replicate
repl = lambda numrep: ak.replicatelayer(numrep)

# Merge
merl = lambda numbranch: ak.mergelayer(numbranch)

# Addition layer
adl = lambda numinp: ak.addlayer(numinp, dim=2, issequence=False)


# 2x2 Upscale layer with interpolation
iusl = lambda us=(2, 2): nk.upsamplelayer(us=list(us), interpolate=True)


def cld(fmapsin, fmapsout, kersize, dilation):
    assert kersize[0] == kersize[1]
    kersize = kersize[0]
    inl = linp(fmapsin)
    out = lcld(inl, fmapsout, kersize, dilation)
    lay = ak.lasagnelayer(inputlayers=inl, outputlayers=out)
    return lay


# Global pool layer
def gpl(fmapsin):
    inp = ll.InputLayer((None, fmapsin, None, None))
    gp = ll.GlobalPoolLayer(inp)
    rs = ll.ReshapeLayer(gp, (-1, fmapsin, 1, 1))
    lay = ak.lasagnelayer(inputlayers=inp, outputlayers=rs)
    return lay


# Define residual blocks
def resblock(fmapsin, kersize=3):
    fmapsmid = (fmapsin * 3)/4
    stream = acl() + cl(fmapsin, fmapsmid, [kersize, kersize]) + cld(fmapsmid, fmapsmid, [kersize, kersize], 2) + \
             cll(fmapsmid, fmapsin, [1, 1])
    block = repl(2) + idl() * stream + adl(2)
    return block


def terminate(mode):
    if mode == 'upsample':
        term = iusl() + cl(32, 32, [3, 3]) + cl(32, 16, [3, 3]) + cl(16, 16, [3, 3]) + gpl(16)
    elif mode == 'downsample':
        term = scl(32, 64, [3, 3]) + scl(64, 128, [3, 3]) + scl(128, 256, [3, 3]) + scl(256, 256, [3, 3]) + \
               scl(256, 256, [3, 3]) + scl(256, 256, [3, 3]) + scl(256, 256, [3, 3]) + scl(256, 256, [3, 3]) + \
               clv(256, 256, [4, 4]) + cl(256, 128, [1, 1]) + cl(128, 64, [1, 1]) + cl(64, 32, [1, 1]) + cl(32, 16, [1, 1]) + cl(16, 16, [3, 3])
    else:
        raise NotImplementedError
    return term


def build(numinp=3, numout=1, parampath=None, finalactivation='sigmoid', subdepth=2, blockconfig=None):

    if finalactivation == 'sigmoid':
        fl = cls
    elif finalactivation == 'linear':
        fl = cll
    else:
        raise NotImplementedError

    if blockconfig is None:
        blockconfig = [(32, 5), (64, 5), (128, 3), (64, 3), (32, 3)]

    # Build network
    net = scll(numinp, 32, [7, 7]) + \
          reduce(lambda x, y: x + y, [resblock(*blockconfig[0]) for _ in range(subdepth)]) + cll(32, 64, [1, 1]) + \
          reduce(lambda x, y: x + y, [resblock(*blockconfig[1]) for _ in range(subdepth)]) + cll(64, 128, [1, 1]) + \
          reduce(lambda x, y: x + y, [resblock(*blockconfig[2]) for _ in range(subdepth)]) + cll(128, 64, [1, 1]) + \
          reduce(lambda x, y: x + y, [resblock(*blockconfig[3]) for _ in range(subdepth)]) + cll(64, 32, [1, 1]) + \
          reduce(lambda x, y: x + y, [resblock(*blockconfig[4]) for _ in range(subdepth)]) + \
          repl(2) + terminate('upsample') * terminate('downsample') + merl(2) + cl(32, 32, [1, 1]) + \
          cl(32, 32, [1, 1]) + cl(32, 16, [1, 1]) + cl(16, 8, [1, 1]) + fl(8, numout, [1, 1])

    # Load parameters
    if parampath is not None:
        net.load(parampath)

    # Set up learning rate
    net.baggage["learningrate"] = th.shared(value=np.float32(0.0002))

    # Last layer: no regularization
    nu.setbaggage(net[-1].params, regularizable=False)

    # Done
    return net

if __name__ == '__main__':
    net = build(4, 1)
    pass
