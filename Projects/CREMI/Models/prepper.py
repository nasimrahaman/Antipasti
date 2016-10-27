"""Prepare model for training."""

import numpy as np
import theano as th
import theano.tensor as T


# Prepare network
def prep(net, optimizer='adam', initlearningrate=0.0002, savedir=None, parampath=None):

    # Load parameters
    if parampath is not None:
        net.load(parampath)
    # Set l2 coefficient
    net.baggage['l2'] = th.shared(np.float32(0.0001))
    # Compute loss
    net.cost(method='bce')
    # Set learningrate
    net.baggage['learningrate'] = th.shared(np.float32(initlearningrate))
    # Make optimizer
    net.getupdates(method=optimizer, learningrate=net.baggage['learningrate'])
    # Fix broadcasting
    broadcastparams(net)

    # Set save directory
    if savedir is not None:
        net.savedir = savedir

    # Done
    return net


def broadcastparams(net):
    # Changes need to be persistent
    for n in range(len(net.updates)):
        if net.updates[n][0].broadcastable != net.updates[n][1].broadcastable:
            # Figure out which axes to broadcast
            axes = [axisnum for axisnum, axis in enumerate(net.updates[n][0].broadcastable) if axis]
            # Broadcast
            net.updates[n][1] = T.addbroadcast(net.updates[n][1], *axes)

