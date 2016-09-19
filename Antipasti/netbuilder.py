__author__ = "Nasim Rahaman"

__doc__ = \
"""This file contains convenience wrappers for building and editing networks"""

import netkit as nk
import netarchs as na


def cat(*nets):
    """
    Concatenate two or more networks (or layers) provided as arguments

    :type nets: list of netkit.layer or list of netkit.layertrain
    :param nets: Networks to concatenate

    :rtype: netarchs.layertrain
    """
    assert all([isinstance(net, (nk.layer, na.layertrain)) for net in nets]), \
        "Networks must be a layer or layertrain instance."

    return sum(nets)


def addoutputmaps(layer, numnewmaps):
    """
    Add more output maps to layer. Uses a simplified Net2WiderNet approach.

    :type layer: netkit.layer
    :param layer: Layer to add output maps to.

    :type numnewmaps: int
    :param numnewmaps: Number of new output maps to add.

    :rtype: netkit.layer
    """

    assert isinstance(layer, nk.layer), "Layer must be an Antipasti layer."

    return layer * int(numnewmaps)


def addinputmaps(layer, numnewmaps):
    """
    Add more input maps to layer.

    :type layer: netkit.layer
    :param layer: Layer to add input maps to.

    :type numnewmaps: int
    :param numnewmaps: Number of new input maps to add.

    :rtype: netkit.layer
    """

    assert isinstance(layer, nk.layer), "Layer must be an Antipasti layer."

    return layer / int(numnewmaps)


def catoutputmaps(layers):
    """
    Concatenate the output maps of layers 1 and 2 (or more). If layer1 is a layer with (say) 20 input and 30 output maps
    and layer2 one with 20 input and 50 output maps, the resulting layer would have 20 input and 80 output maps,
    wherein output maps 0-29 belong to layer1 and 30-79 to layer2.

    :type layers: list of netkit.layer
    :param layers: List of layers to concatenate.

    :rtype: netkit.layer
    """

    assert all([isinstance(layer, nk.layer) for layer in layers]), "Layers must be Antipasti layers."

    return reduce(lambda x, y: x * y, layers)


def newlayer(incoming, layertype, *args, **kwargs):
    """
    Build a network like you would in Lasagne.
    Example usage:
        conv1 = newlayer(None, convlayer, fmapsin=1, fmapsout=20, kersize=[5, 5])
        pool1 = newlayer(conv1, poollayer, ds=[2, 2])
        conv2 = newlayer(pool1, convlayer, fmapsin=20, fmapsout=30, kersize=[3, 3])
        softmax1 = newlayer(conv2, softmaxlayer, dim=2)
        net = softmax1

    :type incoming: netkit.layer or netarchs.layertrain
    :param incoming: Incoming layer

    :type layertype: str or netkit.layer
    :param layertype: Next layer to add

    :type args: list
    :param args: Arguments to initialize layer with

    :type kwargs: dict
    :param kwargs: Keyword arguments to initialize layer with

    :rtype: netarchs.layertrain
    """

    assert incoming is None or isinstance(incoming, (nk.layer, na.layertrain)), "Incoming must be an Antipasti layer " \
                                                                                "or a network (layertrain) or None."
    assert isinstance(layertype, str) or issubclass(layertype, nk.layer), \
        "layertype must be an Antipasti layer or a string."

    layertype = getattr(nk, layertype) if isinstance(layertype, str) else layertype

    if incoming is None:
        return layertype(*args, **kwargs)
    else:
        return incoming + layertype(*args, **kwargs)

