__author__ = "nasimrahaman"

""" Macros """

import theano as th
import numpy as np
import theano.tensor as T

import Antipasti.netkit as nk
import Antipasti.netarchs as na
import Antipasti.netdatakit as ndk
import Antipasti.netutils as nu
import Antipasti.prepkit as pk


# Function for layerwise pretraining of layertrain (optionally followed by finetuning)
def layerwisepretrain(ltrain, trX, l2train=None, maxiterperlayer=75, maxiterfinetune=75, maxtime=np.inf):
    """
    :type ltrain: na.layertrain
    :param ltrain: The whole layer train

    :type l2train: list
    :param l2train: List of slices of the layertrain to train

    :type maxiterperlayer: int
    :param maxiterperlayer: Maximum number of iterations per layer

    :type maxiterfinetune: int
    :param maxiterfinetune: Maximum number of iterations for fine tuning (set to 0 to skip)

    :type maxtime: float or int
    :param maxtime: Maximum allocated time

    :return: Pretrained layertrain
    """

    # Parse
    if l2train is None:
        l2train = [slice(k, k + 1) for k in range(len(ltrain))]

    # Main loop over layers
    for layernums in l2train:
        # Step 0: Get rid of irrelevant layer parameters in ltrain by setting activetrain
        ltrain.activetrain = ltrain.train[layernums]
        # Check if there are any trainable parameters at all
        if len(ltrain.params) == 0:
            continue

        # Step 1: Deactivate what's not needed
        # Say layernums = 2:3. Then:
        #
        # [1e] >> [2e] >> [3e] >> [4e] >> [4d] >> [3d] >> [2d] >> [1d]
        # [1e] >> [2e] >> [3e] >> [xx] >> [xx] >> [3d] >> [xx] >> [xx]
        #                  ^                       ^
        #                 soll                    ist

        # Deactivate all layers after layernum
        for layer in ltrain.train[layernums.stop:-1]:
            layer.deactivate('all')

        # Activate encoder and decoder for layernum
        for layer in ltrain.train[layernums]:
            layer.activate('all')

        # Deactivate decoder and activate encoder for layers before layernum
        for layer in ltrain.train[0:layernums.start]:
            layer.deactivate('dec')
            layer.activate('enc')

        # Step 2: Assign ist and soll
        ist = ltrain.trainer.model.train[layernums][0].xr
        soll = ltrain.trainer.model.train[layernums][0].x

        # Print
        print("Training Layer(s): {} - {}".format(layernums.start, layernums.stop - 1))
        # Train
        ltrain.trainer.train(trX=trX, ist=ist, soll=soll, maxiter=maxiterperlayer, maxtime=maxtime)

    # Fine tuning
    # Step 0: Reset parameter list
    ltrain.activetrain = ltrain.train

    # Step 1: Activate all encoders and decoders
    for layer in ltrain.train:
        layer.activate('all')

    # Train
    print("Fine tuning all layers...")
    ltrain.trainer.train(trX=trX, maxiter=maxiterperlayer, maxtime=maxtime)

    return ltrain
