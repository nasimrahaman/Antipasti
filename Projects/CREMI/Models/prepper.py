"""Prepare model for training."""

import numpy as np
import theano as th
import theano.tensor as T

import Antipasti.netrain as nt


# Prepare network
def prep(net, optimizer='adam', initlearningrate=0.0002, savedir=None, parampath=None):

    # Load parameters
    if parampath is not None:
        net.load(parampath)
    # Set l2 coefficient
    net.baggage['l2'] = th.shared(np.float32(0.0001))
    net.baggage['wt'] = th.shared(np.float32(0.5))
    # Compute loss
    net.cost(method=wcbce, wt=net.baggage['wt'], regterms=[(2, net.baggage['l2'])])
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
            net.updates[n] = (net.updates[n][0], T.addbroadcast(net.updates[n][1], *axes))


def wcbce(ist, soll, wt):
    # Weight controlled binary cross entropy
    ist, soll = nt.prep(ist, soll, clip=True)
    # Compute weighted loss
    L = -T.mean(T.sum(wt * soll * T.log(ist) + (1. - wt) * (1. - soll) * T.log(1. - ist), axis=1))
    # Done
    return L
