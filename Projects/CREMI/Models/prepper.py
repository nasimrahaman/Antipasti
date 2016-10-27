"""Prepare model for training."""

import numpy as np
import theano as th


# Prepare network
def prep(net, optimizer='adam', initlearningrate=0.0002, savedir=None, parampath=None):

    # Load parameters
    if parampath is not None:
        net.load(parampath)

    # Compute loss
    net.cost(method='bce')
    # Set learningrate
    net.baggage['learningrate'] = th.shared(np.float32(initlearningrate))
    # Make optimizer
    net.getupdates(method=optimizer, learningrate=net.baggage['learningrate'])

    # Set save directory
    if savedir is not None:
        net.savedir = savedir

    # Done
    return net
