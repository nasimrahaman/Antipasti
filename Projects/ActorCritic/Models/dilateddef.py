
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

# Convlayer without activation
cll = lambda fmapsin, fmapsout, kersize: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout, kersize=kersize)

# Convlayer with Sigmoid
cls = lambda fmapsin, fmapsout, kersize: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout, kersize=kersize,
                                                      activation=ntl.sigmoid())


def cld(fmapsin, fmapsout, kersize, dilation):
    assert kersize[0] == kersize[1]
    kersize = kersize[0]
    inl = linp(fmapsin)
    out = lcld(inl, fmapsout, kersize, dilation)
    lay = ak.lasagnelayer(inputlayers=inl, outputlayers=out)
    return lay


# Build Model
def build(numinp=3, numout=3, parampath=None, finalactivation='sigmoid', applylastlayerl2=False):

    if finalactivation == 'sigmoid':
        fl = cls
    elif finalactivation == 'linear':
        fl = cll
    else:
        raise NotImplementedError

    # Make network
    net = cl(numinp, 32, [5, 5]) + \
          cld(32, 64, [5, 5], 2) + cld(64, 128, [5, 5], 4) + cld(128, 64, [5, 5], 2) + \
          cl(64, 32, [5, 5]) + \
          fl(32, numout, [1, 1])

    if not applylastlayerl2:
        # Make the last layer unregularizable
        nu.setbaggage(net[-1].params, regularizable=False)

    # Load parameters
    if parampath is not None:
        net.load(parampath)

    # Set up learning rate
    net.baggage["learningrate"] = th.shared(value=np.float32(0.0002))
    # Done
    return net
