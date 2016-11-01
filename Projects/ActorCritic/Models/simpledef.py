
import theano as th
import numpy as np

import Antipasti.netkit as nk
import Antipasti.netarchs as na
import Antipasti.archkit as ak
import Antipasti.netools as ntl

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


def build(numinp=3, numout=3, parampath=None, finalactivation='sigmoid'):

    if finalactivation == 'sigmoid':
        fl = cls
    elif finalactivation == 'linear':
        fl = cll
    else:
        raise NotImplementedError

    # Build simple as fuck network
    net = scl(numinp, 32, [5, 5]) + cl(32, 32, [3, 3]) + usl() + fl(32, numout, [3, 3])
    # Load parameters
    if parampath is not None:
        net.load(parampath)
    # Set up learning rate
    net.baggage["learningrate"] = th.shared(value=np.float32(0.0002))
    # Done
    return net
