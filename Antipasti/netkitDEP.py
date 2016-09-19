import Antipasti.netools

__author__ = 'Nasim Rahaman'

''' Class definitions for building a modular CNN'''

# DONE: Define classes:
#   convlayer
#   poollayer
#   unpoollayer
#   mlplayer
#   noiselayer

# TODO: Implement Dropout
# TODO: Implement Noise Layer
# TODO: Implement Decoder Convolution method in convlayer

# Global Imports
import theano as th
import theano.tensor as T
import theano.tensor.nnet.conv as conv
import theano.tensor.nnet.conv3d2d as conv3d2d
from theano.tensor.signal import downsample
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import netutils


class convlayer:

    """ General Convolutional Layer """

    # Constructor
    def __init__(self, fmapsin, fmapsout, kersize, activation=Antipasti.netools.relu, noise=0, alpha=1, makedecoder=False,
                 W=None, b=None, bp=None):

        """
        :type fmapsin: int
        :param fmapsin: Number of input feature maps

        :type fmapsout: int
        :param fmapsout: Number of output feature maps

        :type kersize: 2- or 3-tuple
        :param kersize: size of the convolution kernel; A 2-tuple (3-tuple) initializes a 2D (3D) conv. layer

        :type activation: theano symbolic function
        :param activation: transfer function of the layer

        :type noise: float between 0 and 1
        :param noise: (meta) parameter to determine the extent to which the input is noised

        :type alpha: float
        :param alpha: Initialization gain (W ~ alpha * N(0, 1))

        :type makedecoder: bool
        :param makedecoder: Boolean switch for initializing decoder biases

        :type W: theano tensor of size (fmapsout, kersize[2], fmapsin, kersize[0], kersize[1]) in 3D,
                                       (fmapsout, fmapsin, kersize[0], kersize[1]) in 2D
        :param W: Preset weight tensor of the layer (use for tying weights)

        :type b: theano vector of size (fmapsout,)
        :param b: Preset bias vector of the layer (use for tying weights)

        :type bp: theano vector of size (fmapsin,)
        :param bp: Preset bias vector of the associated decoder layer (use for tying weights)

        :return: None
        """

        # Meta
        self.fmapsin = fmapsin
        self.fmapsout = fmapsout
        self.kersize = kersize
        self.activation = activation
        self.noise = noise
        self.alpha = alpha
        self.decoderactive = makedecoder

        # Parse network dimension
        self.dim = len(self.kersize)

        # Initialize weights W and biases b:
        # W.shape = (fmapsout, kersize[2], fmapsin, kersize[0], kersize[1])     [3D]
        # W.shape = (fmapsout, fmapsin, kersize[0], kersize[1])                 [2D]
        # b.shape = (fmapsout,)

        # Weights
        if not W:
            if self.dim == 3:
                self.W = th.shared(
                    value=np.asarray(
                        self.alpha*np.random.normal(0, 1, size=(fmapsout, kersize[2], fmapsin, kersize[0], kersize[1])),
                        dtype=th.config.floatX),
                    name='W')
            else:
                self.W = th.shared(
                    value=np.asarray(
                        self.alpha*np.random.normal(0, 1, size=(fmapsout, fmapsin, kersize[0], kersize[1])),
                        dtype=th.config.floatX),
                    name='W')
        else:
            self.W = W

        # Biases
        if not b:
            self.b = th.shared(
                value=self.alpha*np.asarray(np.random.normal(0, 1, size=(fmapsout,)), dtype=th.config.floatX),
                name='b')
        else:
            self.b = b

        # Decoder Biases (if decoder active)
        if self.decoderactive:
            if not bp:
              # self.bp = -self.b
                # DEBUG
                self.bp = th.shared(
                    value=self.alpha*np.asarray(np.random.normal(0, 1, size=(fmapsin,)), dtype=th.config.floatX),
                    name='bp')
            else:
                self.bp = bp

        # Fold Parameters
        if self.decoderactive:
            self.params = [self.W, self.b, self.bp]
        else:
            self.params = [self.W, self.b]

        # Container for input (see feedforward() for input shapes) and output
        if self.dim == 2:
            # Input
            self.x = T.tensor('float64', [False, False, False, False], name='x')
            # Output
            self.y = T.tensor('float64', [False, False, False, False], name='y')
            # Reconstructed input
            if self.decoderactive:
                self.xr = T.tensor('float64', [False, False, False, False], name='xr')

        elif self.dim == 3:
            # Input
            self.x = T.tensor('float64', [False, False, False, False, False], name='x')
            # Output
            self.y = T.tensor('float64', [False, False, False, False, False], name='y')
            # Reconstructed input
            if self.decoderactive:
                self.xr = T.tensor('float64', [False, False, False, False, False], name='xr')

        # RNG for Noise
        pass

    # Feed forward through the layer
    def feedforward(self, inp=None, reshape=False, activation=None):
        # Argument inp is expected of the form:
        #    inp.shape = (numimages, z, fmapsin, y, x)     [3D]
        #    inp.shape = (numimages, fmapsin, y, x)        [2D]
        # Setting reshape to True assumes that the 3D input is of the form (numimages, fmapsin, y, x, z)

        # Parse input
        if not inp:
            inp = self.x

        if not activation:
            activation = self.activation

        # Reshape if requested
        if self.dim == 3 and reshape:
            inp = inp.dimshuffle(0, 2, 3, 4, 1)

        # Noise input
        pass

        # Convolve
        if self.dim == 2:
            # IW.shape = (numimages, fmapsout, y, x)
            IW = conv.conv2d(input=inp, filters=self.W, border_mode='full')
            self.y = activation(IW + self.b.dimshuffle('x', 0, 'x', 'x'))
            return self.y

        elif self.dim == 3:
            # IW.shape = (numstacks, z, fmapsout, y, x)
            IW = conv3d2d.conv3d(signals=inp, filters=self.W, border_mode='full')
            self.y = activation(IW + self.b.dimshuffle('x', 'x', 0, 'x', 'x'))
            return self.y

    # Feed forward through the decoder layer (relevant only when used with convolutional autoencoder) [EXPERIMENTAL]
    def decoderfeedforward(self, inp=None, reshape=False, activation=None):
        # Argument inp is expected of the form:
        #   inp.shape = (numimages, z, fmapsout, y, x)      [3D]
        #   inp.shape = (numimages, fmapsout, y, x)         [2D]
        # This layer tries to map a (numimages, z, fmapsout, y, x) image to (numimages, z, fmapsin, y, x).
        # The convolution filters are flipped along the zyx axes.

        # Parse input
        if not inp:
            inp = self.y

        if not activation:
            activation = self.activation

        # Reshape if requested
        if self.dim == 3 and reshape:
            inp = inp.dimshuffle(0, 2, 3, 4, 1)

        if self.dim == 2:
            # Flip conv. kernel
            Wt = self.flipconvfilter()
            # Convolve, transfer and return
            # IW.shape = (numimages, fmapsin, y, x)
            IW = conv.conv2d(input=inp, filters=Wt, border_mode='full')
            self.xr = self.activation(IW + self.bp.dimshuffle('x', 0, 'x', 'x'))
            return self.xr

        elif self.dim == 3:
            # Flip conv. kernel
            Wt = self.flipconvfilter()
            # Convolve, transfer and return
            # IW.shape = (numstacks, z, fmapsin, y, x)
            IW = conv3d2d.conv3d(signals=inp, filters=Wt, border_mode='full')
            self.xr = self.activation(IW + self.bp.dimshuffle('x', 'x', 0, 'x', 'x'))
            return self.xr

    # Method to flip conv. filters for the decoder layer
    def flipconvfilter(self):
        # Flips the convolution filter along zyx axes and shuffles input and output dimensions.
        # Remember,
        # W.shape = (fmapsout, kersize[2], fmapsin, kersize[0], kersize[1])     [3D]
        # W.shape = (fmapsout, fmapsin, kersize[0], kersize[1])                 [2D]

        if self.dim == 2:
            # Shuffle input with output
            Wt = self.W.dimshuffle(1, 0, 2, 3)
            # Flip yx
            Wt = Wt[::, ::, ::-1, ::-1]
            return Wt

        elif self.dim == 3:
            # Shuffle input with output
            Wt = self.W.dimshuffle(2, 1, 0, 3, 4)
            # Flip zyx
            Wt = Wt[::, ::-1, ::, ::-1, ::-1]
            return Wt

    # Method to apply layer parameters
    def applyparams(self, params):
        # if decoder active, params is of the form [W, b, bp] ([W, b] otherwise)

        if self.decoderactive:
            # Unpack
            W, b, bp = params
            # Set
            self.W.set_value(W)
            self.b.set_value(b)
            self.bp.set_value(bp)
        else:
            # Unpack
            W, b = params
            # Set
            self.W.set_value(W)
            self.b.set_value(b)

        pass

    # Trims the edges of the convolution result to compensate for zero padding in full convolution.
    def trim(self, inp=None, numconvs=2):
        # Say: h = x * W (full convolution). h.shape[i] = x.shape[i] + W.shape[i] - 1
        # Remember,
        #   inp.shape = (numimages, z, fmapsout, y, x)      [3D]
        #   inp.shape = (numimages, fmapsout, y, x)         [2D]
        # W.shape = (fmapsout, kersize[2], fmapsin, kersize[0], kersize[1])     [3D]
        # W.shape = (fmapsout, fmapsin, kersize[0], kersize[1])                 [2D]

        # Parse
        if not inp:
            inpgiven = False
            inp = self.xr
        else:
            inpgiven = True

        # 2D
        if self.dim == 2:
            trimy, trimx = [numconvs * (self.W.shape[i] - 1)/2 for i in [2, 3]]     # trimmersize = [trimy, trimx]
            out = inp[::, ::, (trimy):(-trimy), (trimx):(-trimx)]
        # 3D
        elif self.dim == 3:
            trimz, trimy, trimx = [numconvs * (self.W.shape[i] - 1)/2 for i in [1, 3, 4]]
            out = inp[::, (trimz):(-trimz), ::, (trimy):(-trimy), (trimx):(-trimx)]

        # Assign return value (out) to self.xr only if no input given (i.e. inp = self.xr)
        if not inpgiven:
            self.xr = out
            return self.xr
        else:
            return out


class poollayer:

    """ General Max-pooling Layer """

    # Constructor
    def __init__(self, ds):
        """
        :type ds: 2- or 3-tuple for 2 or 3 dimensional network
        :param ds: tuple of downsampling ratios
        """

        # Meta
        self.ds = ds

        # Parse Network Dimensions
        self.dim = len(self.ds)

        # Container for input (see feedforward() for input shapes)
        if self.dim == 2:
            # Input
            self.x = T.tensor('float64', [False, False, False, False], name='x')
            # Output
            self.y = T.tensor('float64', [False, False, False, False], name='y')

        elif self.dim == 3:
            # Input
            self.x = T.tensor('float64', [False, False, False, False, False], name='x')
            # Output
            self.y = T.tensor('float64', [False, False, False, False, False], name='y')

    # Feed forward through the layer (pool)
    def feedforward(self, inp=None, reshape=False):
        # Argument inp is expected of the form:
        #    inp.shape = (numimages, z, fmapsin, y, x)     [3D]
        #    inp.shape = (numimages, fmapsin, y, x)        [2D]
        # Setting reshape to True assumes that the 3D input is of the form (numimages, fmapsin, y, x, z)

        # Parse input
        if not inp:
            inp = self.x

        # Reshape if requested
        if self.dim == 3 and reshape:
            inp = inp.dimshuffle(0, 2, 3, 4, 1)

        # Get poolin'
        if self.dim == 2:
            # Downsample
            self.y = downsample.max_pool_2d(input=inp, ds=self.ds)
            return self.y

        elif self.dim == 3:
            # Theano lesson: downsample.max_pool_2d downsamples the last 2 dimensions in a tensor. To pool in 3D, the z
            # dimension needs to be pooled separately after 'rotating' the tensor appropriately such that the z axis is
            # the last dimension.

            # parse downsampling ratio
            dsyx = self.ds[0:2]
            ds0z = (1, self.ds[2])

            # Dowsnample yx
            H = downsample.max_pool_2d(input=inp, ds=dsyx)
            # Rotate tensor
            H = H.dimshuffle(0, 2, 3, 4, 1)
            # Downsample 0z
            H = downsample.max_pool_2d(input=inp, ds=ds0z)
            # Undo rotate tensor
            self.y = H.dimshuffle(0, 4, 1, 2, 3)
            return self.y

    # Feed forward through the decoder layer (unpool, relevant for convolutional autoencoders)
    def decoderfeedforward(self, inp=None, reshape=False):
        # Argument inp is expected of the form:
        #    inp.shape = (numimages, z, fmapsin, y, x)     [3D]
        #    inp.shape = (numimages, fmapsin, y, x)        [2D]
        # Setting reshape to True assumes that the 3D input is of the form (numimages, fmapsin, y, x, z)

        # Parse input
        if not inp:
            inp = self.y

        # Reshape if requested
        if self.dim == 3 and reshape:
            inp = inp.dimshuffle(0, 2, 3, 4, 1)

        # Get unpoolin'
        if self.dim == 2:
            # Upsample
            H = T.tile(inp, (1, 1) + self.ds)
            return H

        elif self.dim == 3:
            # parse upsampling ratio
            usrat = (1, self.ds[2], 1, self.ds[0:2])
            # Upsample
            H = T.tile(inp, usrat)
            return H


class mlplayer:
    """ General MLP Layer """

    # Constructor
    def __init__(self, sigsin, sigsout, activation=Antipasti.netools.relu, alpha=1, makedecoder=False, W=None, b=None, bp=None):
        """
        :type sigsin: int
        :param sigsin: Number of input signals per neuron

        :type sigsout: int
        :param sigsout: Number of output signals from the layer = num. of neurons in layer

        :type activation: Callable theano symbolic function
        :param activation: Transfer function

        :type alpha: float
        :param alpha: Initialization gain (W ~ alpha*N(0,1))

        :type W: Theano tensor of shape (sigsin, sigsout)
        :param W: Preset weight matrix

        :type b: Theano vector of shape (sigsout)
        :param b: Preset bias vector
        """

        # Meta
        self.sigsin = sigsin
        self.sigsout = sigsout
        self.activation = activation
        self.alpha = alpha
        self.decoderactive = makedecoder

        # Initialize Weights and Biases
        #   W.shape = (sigsin, sigsout)
        #   b.shape = (sigsout)

        # Weights
        if not W:
            self.W = th.shared(
                value=np.asarray(self.alpha*np.random.normal(0, 1, size=(sigsin, sigsout)), dtype=th.config.floatX),
                name='W')
        else:
            self.W = W

        # Biases
        if not b:
            self.b = th.shared(
                value=np.asarray(self.alpha*np.random.normal(0, 1, size=(sigsout,)), dtype=th.config.floatX),
                name='b')
        else:
            self.b = b

        # Decoder Biases
        if self.decoderactive:
            if not bp:
                self.bp = th.shared(
                    value=np.asarray(self.alpha*np.random.normal(0, 1, size=(sigsin,)), dtype=th.config.floatX),
                    name='bp')
            else:
                self.bp = bp

        # Fold parameters
        if self.decoderactive:
            self.params = [self.W, self.b, self.bp]
        else:
            self.params = [self.W, self.b]

        # Containers
        # Input
        self.x = T.dmatrix('x')
        # Output
        self.y = T.dmatrix('y')
        # Reconstructed Input
        self.xr = T.dmatrix('xr')

    # Feedforward through the layer
    def feedforward(self, inp=None, activation=None):
        # inp is expected of the form
        #   inp.shape = (numimages, sigsin)

        # Parse Input
        if not inp:
            inp = self.x

        if not activation:
            activation = self.activation

        # Autoreshape if required
        if not inp.shape[1:] == (self.sigsin,):
            inp = inp.reshape(inp.shape[0], self.sigsin)

        # FFD
        self.y = self.activation(T.dot(inp, self.W) + self.b)

        return self.y

    # Feedforward through the decoder layer
    def decoderfeedforward(self, inp=None, activation=None):

        # Parse input
        if not inp:
            inp = self.y

        if not activation:
            activation = self.activation

        # Autoreshape if required
        if not inp.shape[1:] == (self.sigsout,):
            inp = inp.reshape(inp.shape[0], self.sigsout)

        # FFD
        self.xr = self.activation(T.dot(inp, self.W.T) + self.bp)

        return self.xr

    # Method to apply layer parameters
    def applyparams(self, params):
        # if decoder active, params is of the form [W, b, bp] ([W, b] otherwise)

        if self.decoderactive:
            # Unpack
            W, b, bp = params
            # Set
            self.W.set_value(W)
            self.b.set_value(b)
            self.bp.set_value(bp)
        else:
            # Unpack
            W, b = params
            # Set
            self.W.set_value(W)
            self.b.set_value(b)


class noiselayer:

    """ General Noising Layer """

    # Constructor
    def __init__(self, noisetype=None, sigma=None, n=None, p=None):
        """
        :type noisetype: str
        :param noisetype: Possible keys: 'normal', 'binomial'.
        :type sigma: float
        :param sigma: std for normal noise
        :type n: float
        :param n: n for binomial (salt and pepper) noise
        :type p: float
        :param p: p for binomial (salt and pepper) noise
        :return:
        """

        # Meta
        if not noisetype:
            self.noisetype = 'normal'
        else:
            self.noisetype = noisetype

        if not sigma:
            self.sigma = 1
        else:
            self.sigma = sigma

        if not n:
            self.n = 1
        else:
            self.n = n

        if not p:
            self.p = 0.5
        else:
            self.p = p



        pass


class unpoollayer:

    ''' General Unpooling Layer '''

    def __init__(self, us):
        """
        :type us: tuple of ints
        :param us: upsampling ratio in (y, x, z) in 3D and (y, x) in 2D
        """

        # Meta
        self.us = us

        # Parse Network Dimensions
        self.dim = len(self.us)

        # Container for input (see feedforward() for input shapes)
        if self.dim == 2:
            self.x = T.tensor('float64', [False, False, False, False], name='x')
        elif self.dim == 3:
            self.x = T.tensor('float64', [False, False, False, False, False], name='x')

    def feedforward(self, inp=None, reshape=False):
        # Argument inp is expected of the form:
        #    inp.shape = (numimages, z, fmapsin, y, x)     [3D]
        #    inp.shape = (numimages, fmapsin, y, x)        [2D]
        # Setting reshape to True assumes that the 3D input is of the form (numimages, fmapsin, y, x, z)

        # Parse input
        if not inp:
            inp = self.x

        # Reshape if requested
        if self.dim == 3 and reshape:
            inp = inp.dimshuffle(0, 2, 3, 4, 1)

        # Get unpoolin'
        if self.dim == 2:
            # Upsample
            H = T.tile(inp, (1, 1) + self.us)
            return H

        elif self.dim == 3:
            # parse upsampling ratio
            usrat = (1, self.us[2], 1, self.us[0:2])
            # Upsample
            H = T.tile(inp, usrat)
            return H
