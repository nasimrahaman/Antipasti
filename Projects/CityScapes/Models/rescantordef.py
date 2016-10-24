"""Defines the residual Cantor."""

import sys
import os

sys.path.append(os.path.abspath('{}/../'.format(__file__)))

import theano as th
import theano.tensor as T
import numpy as np

import Antipasti.netkit as nk
import Antipasti.netarchs as na
import Antipasti.archkit as ak
import Antipasti.netools as ntl
import Antipasti.netrain as nt

import lasagne.layers as ll
import lasagne.nonlinearities as nl

from prepper import prep

# ---- Lasagne
# Convlayer with ELU
lcl = lambda incoming, numout, kersize: ll.Conv2DLayer(incoming, num_filters=numout, filter_size=kersize,
                                                       nonlinearity=nl.elu, pad='same')

lcll = lambda incoming, numout, kersize: ll.Conv2DLayer(incoming, num_filters=numout, filter_size=kersize,
                                                        nonlinearity=nl.linear, pad='same')

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

# ---- Antipasti
initscheme = 'xavier'

# Convlayer with ELU
cl = lambda fmapsin, fmapsout, kersize: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout, kersize=kersize,
                                                     activation=ntl.elu(), W=initscheme)

# Softmax
sml = lambda: nk.softmax(dim=2)

# Convlayer with Sigmoid
cls = lambda fmapsin, fmapsout, kersize: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout, kersize=kersize,
                                                      activation=ntl.sigmoid(), W=initscheme)

# Convlayer without activation
cll = lambda fmapsin, fmapsout, kersize: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout, kersize=kersize, W=initscheme)

# 2x2 Upscale layer with interpolation
iusl = lambda us=(2, 2): nk.upsamplelayer(us=list(us), interpolate=True)

# Identity
idl = lambda: ak.idlayer()

# Parallel tracks
trks = lambda *layers: na.layertrainyard([list(layers)])

# Merge
merl = lambda numbranch: ak.mergelayer(numbranch)

# Dropout layer
drl = lambda p=0.5: nk.noiselayer(noisetype='binomial', p=p)


# Residual Cantor Block.
def block(N, incomings=None, pos='mid', wrap=True):
    # Check if block is wrappable
    wrappable = incomings is None

    # Check if incomings given. If not, make input layer
    if incomings is None:
        if pos == 'start':
            incomings = [linp(8 * N), linp(12 * N), linp(6 * N), linp(3 * N)]
        else:
            # mid or stop
            incomings = [linp(8 * N), linp(8 * N), linp(4 * N), linp(12 * N), linp(2 * N), linp(6 * N), linp(3 * N)]

    if pos == 'start':
        inl3, inl2, inl1, inl0 = incomings
    else:
        inl3r, inl3, inl2r, inl2, inl1r, inl1, inl0 = incomings

    # s: stage (along depth), l: level (resolution). Let the brainfuck begin.

    # Stage 0, pre-residual add.
    s0l3 = lcl(inl3, 8 * N, 3)
    s0l2 = lcl(inl2, 4 * N, 5)
    s0l1 = lcl(inl1, 2 * N, 7)
    s0l0 = lcl(inl0, 1 * N, 9)
    # Stage 1, post-residual add.
    if pos == 'start':
        s0l3r = s0l3
        s0l2r = s0l2
        s0l1r = s0l1
    else:
        s0l3r = ladl(inl3r, s0l3)
        s0l2r = ladl(inl2r, s0l2)
        s0l1r = ladl(inl1r, s0l1)

    # Stage 1
    s1l3 = lcl(lmerl(s0l3r, lspl(s0l2)), 8 * N, 3)

    # Stage 2
    s2l3 = lcl(s1l3, 8 * N, 3)
    s2l3r = ladl(s0l3r, s2l3)
    s2l2 = lcl(lmerl(lusl(s1l3), s0l2r, lspl(s0l1)), 4 * N, 5)

    # Stage 3
    s3l3 = lcl(lmerl(s2l3r, lspl(s2l2)), 8 * N, 3)

    # Stage 4
    s4l3 = lcl(s3l3, 8 * N, 3)
    s4l3r = ladl(s2l3r, s4l3)
    s4l2 = lcl(lmerl(lusl(s3l3), s2l2), 4 * N, 3)
    s4l2r = ladl(s0l2r, s4l2)
    s4l1 = lcl(lmerl(lusl(s2l2), s0l1r, lspl(s0l0)), 2 * N, 7)

    # Stage 5
    s5l3 = lcl(lmerl(s4l3r, lspl(s4l2)), 8 * N, 3)

    # Stage 6
    s6l3 = lcl(s5l3, 8 * N, 3)
    s6l3r = ladl(s4l3r, s6l3)
    s6l2 = lcl(lmerl(lusl(s5l3), s4l2r, lspl(s4l1)), 4 * N, 5)

    # Stage 7
    s7l3 = lcl(lmerl(s6l3r, lspl(s6l2)), 8 * N, 3)

    # Stage t (terminal)
    stl3r = s6l3r
    stl3 = s7l3
    stl2r = s4l2r
    stl2 = lmerl(lusl(s7l3), s6l2)
    stl1r = s0l1r
    stl1 = lmerl(lusl(s6l2), s4l1)
    stl0 = lmerl(lusl(s4l1), s0l0)

    # Done.
    if pos == 'stop':
        outgoings = [stl3, stl2, stl1, stl0]
    else:
        outgoings = [stl3r, stl3, stl2r, stl2, stl1r, stl1, stl0]

    # Wrap up
    if wrap:
        # Can only wrap if incomings is None (i.e. init with input layers)
        assert wrappable
        layer = ak.lasagnelayer(incomings, outgoings)
        return layer
    else:
        return outgoings


# VGG initiator module
def vgginitiate(N=64, parampath=None, trainable=False, lr=None, activation=None):
    import vgg16
    # Build
    start = vgg16.build(parampath=parampath, trainable=trainable, lr=lr, activation=activation)
    # If N != 64, we're gonna need a module to reduce the number of filtermaps.
    if N != 64:
        launch = trks(cl(512, 8 * N, [3, 3]), cl(768, 12 * N, [5, 5]), cl(384, 6 * N, [7, 7]), cl(192, 3 * N, [9, 9]))
    else:
        launch = trks(idl(), idl(), idl(), idl())
    # Return
    return start + launch


# Gradually terminate the cantor chain
def gterminate(numout=19, finalactivation=None, N=30):

    # Parse final layer
    if finalactivation == 'softmax' or finalactivation is None:
        fl = cll(numout, numout, [1, 1]) + sml()
    elif finalactivation == 'linear':
        fl = cll(numout, numout, [1, 1])
    elif finalactivation == 'sigmoid':
        fl = cls(numout, numout, [1, 1])
    else:
        raise NotImplementedError

    fuse = trks(cl(8*N, 4*N, [3, 3]) + iusl(), cl(12*N, 6*N, [3, 3]), cl(6*N, 3*N, [5, 5]), cl(3*N, 2*N, [5, 5])) + \
           trks(merl(2), idl(), idl()) + \
           trks(cl(10*N, 5*N, [3, 3]) + cl(5*N, 3*N, [3, 3]) + iusl(), cl(3*N, 3*N, [3, 3]), cl(2*N, N, [5, 5])) + \
           trks(merl(2), idl()) + \
           trks(cl(6*N, 4*N, [3, 3]) + cl(4*N, 2*N, [3, 3]) + iusl(), cl(N, N, [3, 3])) + \
           merl(2) + \
           cl(3*N, 2*N, [3, 3]) + cl(2*N, N, [3, 3]) + \
           drl() + cl(N, N, [1, 1]) + drl() + cl(N, N, [1, 1]) + drl() + cl(N, numout, [1, 1]) + fl

    return fuse


def build(N=30, depth=5, vggparampath=None, vggtrainable=False, vgglr=None, vggactivation=None, usewmap=True,
          finalactivation='softmax', savedir=None, parampath=None):

    # Hardcode cityscapes
    numinp = 3
    numout = 19

    print("[+] Building Cantor Network of depth {} and base width {} with 3 inputs and 19 outputs.".format(depth, N))

    if vggtrainable:
        assert vgglr is not None
        print("[+] VGG Initiator will be fine tuned with a learningrate {}.".format(vgglr))

    # Wrap vgglr in a theano shared variable (if it's a float)
    vgglr = th.shared(value=np.float32(vgglr)) if vgglr is not None else None

    # Initiator
    init = lambda numinp: vgginitiate(N, parampath=vggparampath, trainable=vggtrainable,
                                      lr=vgglr, activation=vggactivation)
    # Terminator (bam!)
    term = gterminate

    # Build network
    net = init(numinp) + block(N, pos='start') + \
          reduce(lambda x, y: x + y, [block(N, pos='mid') for _ in range(depth - 2)]) + \
          block(N, pos='stop') + term(numout, finalactivation)

    # Add VGG learning rate to baggage to control it externally
    net.baggage['vgg-learningrate'] = vgglr

    net.feedforward()

    net = prep(net, parampath=parampath, usewmap=usewmap, savedir=savedir, optimizer='adam',
               lasagneoptimizer=False, lasagneobj=True)

    return net

if __name__ == '__main__':
    net = build()
