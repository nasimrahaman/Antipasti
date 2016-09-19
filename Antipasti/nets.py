import Antipasti.netools

__author__ = 'nasimrahaman'

''' Network Classes Built on Netkit '''

# TODO: Define classes for:
# CAE Stack (CAES)
# CAE with Embedded Semantic Context (CAESC)

# Imports
import theano as th
import theano.tensor as T
import numpy as np
import Antipasti.netkit as nk
import Antipasti.netutils as nu
from re import split
from warnings import warn


# Convolutional Autoencoder Stack
class net:
    # Constructor
    def __init__(self, configstring):
        """
        :type configstring: str
        :param configstring: Configuration string for the convolutional autoencoder stack.

                             Keywords: Explanation

                              inp : input layer.

                              conv-N@mxnxp:activation : convolutional layer of N feature maps with kernelsize m x n x p
                                                        and transfer function activation. Using just mxn instantiates a
                                                        2d conv. layer. Supported activations: 'relu', 'sigmoid'.
                              pool-mxnxp : pooling layer with ds ratio m x n x p. Using just mxn instatiates a 2d pool
                                           layer.
                              mlp-N:activation : fully connected layer with N neurons and transfer function activation.
                              dec : decoder layers

                             Usage Example:
                             inp -> conv-10@15x15x15:relu -> pool-2x2x2 -> conv-5@10x10x10:relu -> pool-3x3x3 -> dec
        :return:
        """

        # Parse configuration string to layertrain
        self.layertrain = self.parseconfigstring(configstring)

    # LOW PRIORITY
    # Train the CAE layer-wise
    def trainae(self, trX):
        """
        :type trX: float
        :param trX: training data of the form trX.shape = (numimages, z, 1, y, x) in 3D or (numimages, 1, y, x) in 2D
        """

        # TODO: Train/condition layer 1 with trX
        # TODO: Compile feedforward and encode trX by layer 1 encoder
        # TODO: Train/condition layer 2 with trX encoded by layer 2
        # TODO: Repeat for all layers
        pass

    # HIGH PRIORITY
    # Train a layer (= layertrain[layernum]) in the autoencoder
    def trainlayer(self, trX, layernum):
        """
        :type trX: float
        :param trX: training data of the form trX.shape = (numimages, z, fmapsin, y, x) in 3D or
                    (numimages, fmapsin, y, x) in 2D
        :type layernum: int
        :param layernum: index of the layer in layertrain
        """

        # TODO: Configure memory mapping for trX
        # TODO: Feedforward (symbolically) through the network
        # TODO: Compute Cost and Gradient (use helper function)
        # TODO: Compile trainer function [cost, params] = trainer(batch)
        # TODO: Batch gradient descent on trX (use helper function)
        # TODO: Get best parameters and apply to layer

        # Extract layer and check if convolutional
        layer = self.layertrain[layernum]
        if not (isinstance(layer, nk.convlayer) or isinstance(layer, nk.mlplayer)):
            raise NameError('Layer not convolutional/mlp. Check layernum and retry.')

        # Feedforward (symbolically) through the network
        # Encode. The input is contained in layer as layer.x, and so is the output.
        layer.feedforward()

        # Decode. The reconstruction is stored in layer as layer.xr
        layer.decoderfeedforward()

        # Compute Cost and Gradient
        C, dC = self.cost(layer=layer)









        pass

    # HIGH PRIORITY
    # Condition layer by segmentation
    def conditionlayer(self, trXi, layernum, nummapsperclass=None):
        """
        :type trXi: list of numpy ndarrays
        :param trXi: trXi[n] gives the training input for the n-th class. trXi[n].shape = (numimages, z, fmapsin, y, x)
        :type layernum: int
        :param layernum: layer to train in layertrain
        :type nummapsperclass: list or int
        :param nummapsperclass: number of maps per given class
        """

        # TODO: Compute number of maps per class if required
        # TODO: Configure memory mapping
        # TODO: Feedforward (symbolically) through the network
        # TODO: Compute cost and gradient (use helper function)
        # TODO: Compile trainer function [cost, params] = trainer(batch) but with special updates: W - d/dW for maps d-
        # TODO: ... esignated to class i and W + d/dW for maps not designated to i.

        pass

    # VERY HIGH PRIORITY
    # Cost and Gradient Functions
    @staticmethod
    def cost(ist=None, soll=None, gradwrt=None, layer=None, method='mse'):
        """
        :type ist: theano symbolic tensor
        :param ist: source (optional if layer is given)
        :type soll: theano symbolic tensor
        :param soll: target (optional if layer is given)
        :type gradwrt: list of theano tensors
        :param gradwrt: computes graident of cost wrt. gradwrt
        :type layer: nk.convlayer or nk.mlplayer
        :param layer: fed forward layer to compute cost on (optional if ist, soll and gradwrt given)
        :type method: str
        :param method: string specifying the method to use
        :return: C, dC (cost and gradient of cost)
        """

        # TODO: Build a switch list of methods to compute cost
        # TODO: For each member in list, compute cost and derivatives

        pass

    # VERY HIGH PRIORITY
    # Batch Gradient Descent for a given trainer function
    @staticmethod
    def batchgradientdescent(trainer, trX, numbatches=None, batchsize=None):
        """
        :type trainer: compiled function
        :param trainer: trainer function of the form cost, parameters = trainer(batch)
        :type trX: trX
        :param trX: training dataset
        :param numbatches:
        :param batchsize:
        :return: bestcost, bestparams
        """

        # TODO: train by batch gradient descent


    # Parses configuration string to a layer train (choo choo)
    @staticmethod
    def parseconfigstring(configstring):
        """
        :type configstring: str
        :param configstring: Configuration string of the CAES. See constructor for more documentation.
        :return:
        """
        # Get rid of whitespace
        configstring = configstring.replace(" ", "")

        # Split by arrows to a list of layer configurations
        layerconflist = configstring.split('->')

        # Instantiate Layer Train
        layertrain = []

        # Check if decoders are to be made
        if layerconflist[-1] == 'dec':
            makedecoder = True
        else:
            makedecoder = False

        # Loop over layer configuration list and make layers
        for layerconf in layerconflist:

            # Check if layer convolutional
            if 'conv' in layerconf:

                # Fetch layer parameters
                layerparamlist = split('[-@x:]', layerconf)[1:]

                # Parse Layer Parameters:
                # Cast fmapsout and kersize (this handles both 2 and 3D cases)
                fmapsout, kersize = int(layerparamlist[1]), map(int, layerparamlist[2:-1])

                # Fetch fmapsin
                if len(layertrain) == 0:
                    # layertrain empty, i.e. incoming from input layer
                    fmapsin = 1
                else:
                    # layertrain not empty; find the last convlayer and extract the number of output maps
                    fmapsin = layertrain[
                        np.max(np.where([isinstance(layer, nk.convlayer) for layer in layertrain]))].fmapsout

                # Parse activation
                if layerparamlist[-1] == 'relu':
                    activation = Antipasti.netools.relu
                elif layerparamlist[-1] == 'sigmoid':
                    activation = T.nnet.sigmoid
                else:
                    # Revert to ReLU
                    warn('Activation function not recognized, reverting to ReLU...')
                    activation = Antipasti.netools.relu

                # Construct layer and append to layertrain
                layertrain.append(
                    nk.convlayer(fmapsin, fmapsout, kersize, activation=activation, makedecoder=makedecoder))

            # Check if layer pooler
            if 'pool' in layerconf:
                # Fetch layer parameters
                layerparamlist = split('[-x]', layerconf)

                # Parse layer parameters
                ds = map(int, layerparamlist[1:])

                # Construct layer
                layertrain.append(nk.poollayer(ds))

            # Check if layer mlp
            if 'mlp' in layerconf:

                # Fetch layer parameters
                layerparamlist = split('[-:]', layerconf)

                # Fetch number of neurons in layer
                sigsout = int(layerparamlist[1])

                # Determine number of incoming signals

                # Case 1: First fully connected layer after convolutional and pooling layers:
                if np.where([isinstance(layer, nk.mlplayer) for layer in layertrain])[0].size == 0 and \
                                np.where([isinstance(layer, nk.convlayer) for layer in layertrain])[0].size > 0:

                    # layertrain has at least 1 conv layer. Find the last and assign sigsin
                    sigsin = layertrain[
                        np.max(np.where([isinstance(layer, nk.convlayer) for layer in layertrain]))].fmapsout

                # Case 2: Not the first fully connected layer
                elif np.where([isinstance(layer, nk.mlplayer) for layer in layertrain])[0].size > 0:

                    # Find mlp layer and assign sigsin
                    sigsin = layertrain[
                        np.max(np.where([isinstance(layer, nk.mlplayer) for layer in layertrain]))].sigsout

                # Case 3: First layer in the network
                elif len(layertrain) == 0:

                    # Input layer, ergo
                    sigsin = 1

                else:
                    # What the hell do I know
                    raise NameError('Invalid configuration string or network architecture...')

                # Parse activation
                if layerparamlist[-1] == 'relu':
                    activation = Antipasti.netools.relu
                elif layerparamlist[-1] == 'sigmoid':
                    activation = T.nnet.sigmoid
                else:
                    warn('Activation function not recognized, reverting to ReLU...')
                    activation = Antipasti.netools.relu

                # Construct layer
                layertrain.append(nk.mlplayer(sigsin, sigsout, activation=activation, makedecoder=makedecoder))

        return layertrain