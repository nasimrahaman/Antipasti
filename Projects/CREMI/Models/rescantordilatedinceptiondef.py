"""
Defines the residual Cantor where convolutional units are 1-layer inception modules. The inception towers use
convolutional layers at various dilations.
"""

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


# Multiscale module (Lasagne)
def lmsm3(incoming, numout, kersize):
    # Check if numout is divisible by the number of modules
    assert numout % 3 == 0, "MSM3 requires that the number of outputs be divisible by 3."
    # Get number of outputs from a branch
    branchnumout = numout/3
    # Define module
    # 0 Dilation Branch
    b0d = lcl(incoming, branchnumout, kersize)
    # 2 Dilation Branch
    b2d = lcld(incoming, branchnumout, kersize, dilation=2)
    # 4 Dilation Branch
    b4d = lcld(incoming, branchnumout, kersize, dilation=4)
    # Concatenate all
    out = lmerl(b0d, b2d, b4d)
    # Return
    return out


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
    s0l3 = lmsm3(inl3, 8 * N, 3)
    s0l2 = lmsm3(inl2, 4 * N, 5)
    s0l1 = lmsm3(inl1, 2 * N, 7)
    s0l0 = lmsm3(inl0, 1 * N, 9)
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
    s1l3 = lmsm3(lmerl(s0l3r, lspl(s0l2)), 8 * N, 3)

    # Stage 2
    s2l3 = lmsm3(s1l3, 8 * N, 3)
    s2l3r = ladl(s0l3r, s2l3)
    s2l2 = lmsm3(lmerl(lusl(s1l3), s0l2r, lspl(s0l1)), 4 * N, 5)

    # Stage 3
    s3l3 = lmsm3(lmerl(s2l3r, lspl(s2l2)), 8 * N, 3)

    # Stage 4
    s4l3 = lmsm3(s3l3, 8 * N, 3)
    s4l3r = ladl(s2l3r, s4l3)
    s4l2 = lmsm3(lmerl(lusl(s3l3), s2l2), 4 * N, 3)
    s4l2r = ladl(s0l2r, s4l2)
    s4l1 = lmsm3(lmerl(lusl(s2l2), s0l1r, lspl(s0l0)), 2 * N, 7)

    # Stage 5
    s5l3 = lmsm3(lmerl(s4l3r, lspl(s4l2)), 8 * N, 3)

    # Stage 6
    s6l3 = lmsm3(s5l3, 8 * N, 3)
    s6l3r = ladl(s4l3r, s6l3)
    s6l2 = lmsm3(lmerl(lusl(s5l3), s4l2r, lspl(s4l1)), 4 * N, 5)

    # Stage 7
    s7l3 = lmsm3(lmerl(s6l3r, lspl(s6l2)), 8 * N, 3)

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


# Vanilla initiator module with inception and dilated convolutions.
def initiate(numinp=3, N=30, incoming=None, wrap=True):

    wrappable = incoming is None

    if incoming is None:
        incoming = ll.InputLayer((None, numinp, None, None))

    # Number of outputs from a the base branch
    basebranchnumout = N/3

    # How to parse:
    # ilXbY = Initiator Level X; Branch Y.
    # Examples:
    #   il1b0 = Initiator Level 1; Branch 0
    #   il2b3 = Initiator Level 2; Branch 3

    # Level 0
    # Branches
    il0b0 = lcld(lcl(incoming, basebranchnumout, 9), basebranchnumout, 9, 8)
    il0b1 = lcld(lcl(incoming, basebranchnumout, 5), basebranchnumout, 5, 4)
    il0b2 = lcld(lcl(incoming, basebranchnumout, 3), basebranchnumout, 3, 2)
    il0bx = lmerl(il0b0, il0b1, il0b2)
    il0 = lspl(il0bx)

    # Level 1
    # Branches
    il1b0 = lcld(lcl(il0, 2 * basebranchnumout, 7), 2 * basebranchnumout, 7, 6)
    il1b1 = lcld(lcl(il0, 2 * basebranchnumout, 5), 2 * basebranchnumout, 5, 4)
    il1b2 = lcld(lcl(il0, 2 * basebranchnumout, 3), 2 * basebranchnumout, 3, 2)
    il1bx = lmerl(il1b0, il1b1, il1b2)
    il1 = lspl(il1bx)

    # Level 2
    # Branches
    il2b0 = lcld(lcl(il1, 4 * basebranchnumout, 5), 4 * basebranchnumout, 5, 4)
    il2b1 = lcld(lcl(il1, 4 * basebranchnumout, 5), 4 * basebranchnumout, 5, 2)
    il2b2 = lcl(lcl(il1, 4 * basebranchnumout, 5), 4 * basebranchnumout, 3)
    il2bx = lmerl(il2b0, il2b1, il2b2)
    il2 = lspl(il2bx)

    # Level 3
    # Branches
    il3b0 = lcld(lcl(il2, 8 * basebranchnumout, 5), 8 * basebranchnumout, 3, 2)
    il3b1 = lcld(lcl(il2, 8 * basebranchnumout, 3), 8 * basebranchnumout, 3, 2)
    il3b2 = lcl(lcl(il2, 8 * basebranchnumout, 3), 8 * basebranchnumout, 3)
    il3bx = lmerl(il3b0, il3b1, il3b2)
    il3 = il3bx

    # Build exits
    exl0 = lmerl(il0bx, lusl(il1bx))
    exl1 = lmerl(il1bx, lusl(il2bx))
    exl2 = lmerl(il2bx, lusl(il3bx))
    exl3 = il3

    outgoings = [exl3, exl2, exl1, exl0]

    # Wrap if required
    if wrap:
        assert wrappable
        lay = ak.lasagnelayer(inputlayers=incoming, outputlayers=outgoings)
        return lay
    else:
        return outgoings


# Build a terminator module
def terminate(numout=3, N=30, incomings=None, wrap=True):

    wrappable = incomings is None

    if incomings is None:
        incomings = [linp(8 * N), linp(12 * N), linp(6 * N), linp(3 * N)]

    # Unpack incomings to individual inputs
    inl3, inl2, inl1, inl0 = incomings

    Nby3 = N/3
    Nby2 = N/2

    # Level 3
    # Branches
    tl3b0 = lcld(lcl(inl3, 4 * N, 5), 4 * Nby3, 3, 2)
    tl3b1 = lcld(lcl(inl3, 4 * N, 3), 4 * Nby3, 3, 2)
    tl3b2 = lcl(lcl(inl3, 4 * N, 3), 4 * Nby3, 3)
    tl3bx = lmerl(tl3b0, tl3b1, tl3b2)
    tl3 = lusl(tl3bx)

    # Level 2
    # Branches
    tl2b0 = lcld(lcl(lmerl(tl3, inl2), 8 * N, 5), 8 * Nby3, 5, 4)
    tl2b1 = lcld(lcl(lmerl(tl3, inl2), 8 * N, 5), 8 * Nby3, 5, 2)
    tl2b2 = lcl(lcl(lmerl(tl3, inl2), 8 * N, 3), 8 * Nby3, 3)
    tl2bx = lmerl(tl2b0, tl2b1, tl2b2)
    tl2 = lusl(tl2bx)

    # Level 1
    # Branch
    tl1b0 = lcld(lcl(lmerl(tl2, inl1), 7 * N, 7), 7 * Nby3, 7, 6)
    tl1b1 = lcld(lcl(lmerl(tl2, inl1), 7 * N, 5), 7 * Nby3, 5, 4)
    tl1b2 = lcld(lcl(lmerl(tl2, inl1), 7 * N, 3), 7 * Nby3, 3, 2)
    tl1bx = lmerl(tl1b0, tl1b1, tl1b2)
    tl1 = lusl(tl1bx)

    # Level 0
    # Branches
    tl0b0 = lcl(lcl(lmerl(tl1, inl0), 5 * N, 9), 5 * Nby3, 3)
    tl0b1 = lcl(lcl(lmerl(tl1, inl0), 5 * N, 5), 5 * Nby3, 5)
    tl0b2 = lcl(lcl(lmerl(tl1, inl0), 5 * N, 3), 5 * Nby3, 3)
    tl0bx = lmerl(tl0b0, tl0b1, tl0b2)
    tl0 = tl0bx

    # Final LayerS
    # Module A
    flsAb0 = lcl(tl0, 3 * Nby2, 1)
    flsAb1 = lcl(tl0, 3 * Nby2, 3)
    flsA = lmerl(flsAb0, flsAb1)

    # Module B
    flsBb0 = lcl(lcl(flsA, 2 * N, 3), 2 * Nby2, 1)
    flsBb1 = lcl(lcl(flsA, 2 * N, 1), 2 * Nby2, 3)
    flsB = lmerl(flsBb0, flsBb1)

    # Module C
    flsCb0 = lcl(lcl(flsB, 1 * N, 3), 1 * Nby2, 1)
    flsCb1 = lcl(lcl(flsB, 1 * N, 1), 1 * Nby2, 3)
    flsC = lmerl(flsCb0, flsCb1)

    # Module D
    flsD = lcls(flsC, numout, 1)

    if wrap:
        assert wrappable
        lay = ak.lasagnelayer(inputlayers=incomings, outputlayers=flsD)
        return lay
    else:
        return flsD


def build(N=30, depth=5, numinp=3, numout=3, optimizer='adam', savedir=None, parampath=None):

    print("[+] Building Cantor Network of depth {} and base width {} "
          "with {} inputs and {} outputs.".format(depth, N, numinp, numout))

    # Initiator
    init = lambda numinp: initiate(numinp=numinp, N=N)

    # Terminator (bam!)
    term = lambda numout: terminate(numout=numout, N=N)

    # Build network
    net = init(numinp) + block(N, pos='start') + \
          reduce(lambda x, y: x + y, [block(N, pos='mid') for _ in range(depth - 2)]) + \
          block(N, pos='stop') + term(numout=numout)

    # Feedforward
    net.feedforward()

    # Loss and optimizer
    net = prep(net, optimizer=optimizer, savedir=savedir, parampath=parampath)

    # Done.
    return net

if __name__ == '__main__':
    net = build()
