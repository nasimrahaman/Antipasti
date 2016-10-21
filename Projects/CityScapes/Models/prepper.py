"""Prepare an Antipasti network."""

import theano as th
import theano.tensor as T
import numpy as np

import Antipasti.netrain as nt


# Prepare network
def prep(net, parampath=None, optimizer='momsgd', usewmap=True, savedir=None, lasagneoptimizer=False, lasagneobj=False,
         losslevels=None):

    if lasagneoptimizer or lasagneobj:
        import lasagne as las
    else:
        las = None

    # Load params if required to
    if parampath is not None:
        net.load(parampath)
        print("[+] Loaded parameters from {}.".format(parampath))
    else:
        print("[-] Not loading parameters.")

    net.baggage["learningrate"] = th.shared(value=np.float32(0.0002))
    net.baggage["l2"] = th.shared(value=np.float32(0.00001))
    net.baggage["wmap"] = T.tensor4()

    # Set up weight maps if required
    print("[+] Setting up objective...")
    loss(net, usewmap=usewmap, framework=('lasagne' if lasagneobj else 'antipasti'), coalesce=losslevels)

    print("[+] Setting up optimizer with {}...".format("Lasagne" if lasagneoptimizer else "Antipasti"))
    if optimizer == 'momsgd':
        if not lasagneoptimizer:
            net.getupdates(method=optimizer, learningrate=net.baggage["learningrate"], nesterov=True)
        else:
            raise NotImplementedError

    elif optimizer == 'adam':
        if not lasagneoptimizer:
            net.getupdates(method=optimizer, learningrate=net.baggage["learningrate"])
        else:
            net.updates = las.updates.adam(net.dC, net.params, learning_rate=net.baggage['learningrate']).items()

    print("[+] Setting up error metric...")
    # Compute errors
    net = error(net)

    # Assign Save directory.
    if savedir is not None:
        print("[+] Saving weights to {}...".format(savedir))
        net.savedir = savedir
    else:
        print("[-] WARNING: Saving weights to the default directory.")

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


# Compute loss with lasagne
def loss(net, usewmap=True, eps=0.001, coalesce=None, framework='antipasti'):
    print("[+] Setting up objective with {}".format(framework))
    # Compute loss with both frameworks
    if framework == 'lasagne':
        import lasagne as las

        # Flatten to matrices
        ist = T.clip(net.y.dimshuffle(1, 0, 2, 3).flatten(2).dimshuffle(1, 0), eps, 1-eps)
        soll = T.clip(net.yt.dimshuffle(1, 0, 2, 3).flatten(2).dimshuffle(1, 0), eps, 1-eps)
        wmap = net.baggage['wmap'].flatten()

        # Compute loss vector and aggregate with weights
        Lv0 = las.objectives.categorical_crossentropy(ist, soll)
        if usewmap:
            L0 = las.objectives.aggregate(Lv0, wmap)
        else:
            L0 = Lv0.mean()

        # Coalesce loss terms if required
        if coalesce:
            istv1, sollv1 = coalesceclasses(ist, soll, coalesce)
            Lv1 = las.objectives.categorical_crossentropy(istv1, sollv1)
            L1 = las.objectives.aggregate(Lv1, wmap)
            L = L0 + L1
        else:
            L = L0

        # Add L2
        C = L + nt.lp(net.params, regterms=[(2, net.baggage['l2'])])
        # Compute gradients
        dC = T.grad(C, wrt=net.params)
        # Assign to network
        net.L = L
        net.C = C
        net.dC = dC

        return net

    elif framework == 'antipasti':
        if usewmap:
            net.cost(method='cce', wmap=net.baggage['wmap'], regterms=[(2, net.baggage["l2"])], clip=True)
        else:
            net.cost(method='cce', regterms=[(2, net.baggage["l2"])], clip=True)


def coalesceclasses(ist, soll, how):
    assert how
    # ist and soll are flattened vectors.
    gists, gsolls = [], []

    for group in how:
        # Slice
        gist = ist[:, group]
        gsoll = soll[:, group]
        # Sum
        gist = gist.sum(axis=1)
        gsoll = T.clip(gsoll.sum(axis=1), 0., 1.)

        # Append
        gists.append(gist)
        gsolls.append(gsoll)

    # Concatenate to a single tensor
    gists = T.concatenate(gists, axis=1)
    gsolls = T.concatenate(gsolls, axis=1)

    return gists, gsolls
