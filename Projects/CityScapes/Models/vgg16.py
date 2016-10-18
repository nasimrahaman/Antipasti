__doc__ = "Wrap Lasagne VGG-16 in an Antipasti layer."

# VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

import numpy as np
import theano as th

from lasagne.layers import InputLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Upscale2DLayer as UpscaleLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import set_all_param_values
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer

import Antipasti.archkit as ak
import Antipasti.netutils as nu
import Antipasti.netkit as nk
import Antipasti.netarchs as na

from Antipasti.netdatautils import unpickle


def build_model(parampath=None):

    net = {}

    net['input'] = InputLayer((None, 3, None, None))
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False)

    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False)

    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False)

    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False)

    if parampath is not None:
        params = unpickle(parampath)
        set_all_param_values(net['conv4_3'], params['param values'][:-12])

    # Make exit points for the initiator module
    net['exit1'] = net['conv4_3']
    net['exit2'] = ConcatLayer([UpscaleLayer(net['conv4_3'], 2), net['conv3_2']])
    net['exit3'] = ConcatLayer([UpscaleLayer(net['conv3_2'], 2), net['conv2_2']])
    net['exit4'] = ConcatLayer([UpscaleLayer(net['conv2_2'], 2), net['conv1_2']])

    return net


def wrap(model, trainable=False, lr=None):
    lay = ak.lasagnelayer(model['input'], [model['exit1'], model['exit2'], model['exit3'], model['exit4']])
    # Carry the lasagne model dictionary as baggage
    nu.setbaggage(lay, networkdict=model)
    # Set whether parameters are trainable
    nu.setbaggage(lay.params, trainable=trainable)
    # Set learningrate
    if trainable and lr is not None:
        nu.setbaggage(lay.params, learningrate=lr)
    # Build model from layer (not necessary, but wth)
    apmodel = na.layertrainyard([lay])
    return apmodel


def build(parampath=None, trainable=False, lr=None):
    # Build lasagne model with parameters
    lasmodel = build_model(parampath=parampath)
    # Wrap lasagne model
    apmodel = wrap(lasmodel, trainable=trainable, lr=lr)
    # Done
    return apmodel

if __name__ == '__main__':
    build('/home/nrahaman/Downloads/vgg16.pkl')