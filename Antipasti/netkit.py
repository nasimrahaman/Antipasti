__author__ = 'Nasim Rahaman'

''' Class definitions for building a modular CNN'''

# DONE: Define classes:
#   convlayer
#   poollayer
#   unpoollayer
#   mlplayer
#   noiselayer

# Global Imports
import theano as th
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from backend import backend
A = backend()

import numpy as np
from warnings import warn
import netutils
import netools
from netarchs import layertrain
from netarchs import layertrainyard
import theanops as tho
import pykit as pyk
import copy as cp

__doc__ = \
    """
    Readme before extending the Antipasti netkit with new layers!

    To define your own layers,
        - Subclass layer.
        - Write at least the feedforward() and inferoutshape() methods.
        - Any parameters you might need go in the _params attribute. DO NOT ASSIGN TO params!
        - You'll need a parameter connectivity map for every parameter. These go in _cparams. Again, DO NOT ASSIGN TO
          cparams!
        - If you do need to know the input shape to define your parameters, consider using ghost variables (ghostvar).
          See the implementation of batchnormlayer for an example.
        - Recurrent layers require a default circuit, which isn't necessarily easy to understand. Get in touch with me
          (drop by my office or shoot me an email (nasim.rahaman@iwr.uni-heidelberg.de)) and we could talk it through.

    The abstract class "layer" does the rest. Don't forget to initialize a layer object from within your layer's
    __init__ method!
    """


# Master class for feedforward layers
class layer(object):
    """ Superclass for all feedforward layers """

    # Constructor
    def __init__(self):
        # Pre-init duck typed parameters
        self.encoderactive = True
        self.decoderactive = True
        self.numinp = 1
        self.numout = 1
        self.inpdim = None
        self.outdim = None
        self.dim = None
        self.allowsequences = False
        self.issequence = False
        self._testmode = False

        self.layerinfo = None

        self._params = []
        self._cparams = []
        self._state = []
        self.updaterequests = []
        self.getghostparamshape = None

        self._circuit = netutils.fflayercircuit()
        self.recurrencenumber = 0.5

        self._inpshape = None
        self.outshape = None
        self.shapelock = False

        self.x = None
        self.y = None
        self.xr = None

    # inputshape as a property
    @property
    def inpshape(self):
        return self._inpshape

    @inpshape.setter
    def inpshape(self, value):
        # Check if shapelock is armed.
        if self.shapelock:
            warn("Can not set input shape. Disarm shapelock and try again.")
            return

        # Get input shape and set outputshape
        self._inpshape = value
        self.outshape = self.inferoutshape(inpshape=value)
        self.outdim = pyk.delist([len(oshp) for oshp in pyk.list2listoflists(self.outshape)])
        self.numout = pyk.smartlen(self.outdim)

        # Set ghost parameter shapes (if there are any to begin with)
        # FIXME: One may have multiple getghostparamshape's in a layer
        for param, cparam in zip(self.params, self.cparams):
            if isinstance(param, netutils.ghostvar) and callable(self.getghostparamshape):
                param.shape = self.getghostparamshape(value)

            if isinstance(cparam, netutils.ghostvar) and callable(self.getghostparamshape):
                cparam.shape = self.getghostparamshape(value)


    @property
    def testmode(self):
        return self._testmode

    @testmode.setter
    def testmode(self, value):
        self._testmode = value

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = value

    @property
    def cparams(self):
        return self._cparams

    @cparams.setter
    def cparams(self, value):
        self._params = value

    @property
    def circuit(self):
        return self._circuit

    @circuit.setter
    def circuit(self, value):
        raise AttributeError("Not permitted!")

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def isstateful(self):
        return bool(len(self.state))

    # Method to instantiate ghost parameters (if any)
    def instantiate(self):
        """Method to instantiate ghost parameters (if any)."""
        self._params = [param.instantiate() if isinstance(param, netutils.ghostvar) and param.instantiable else
                        param for param in self._params]
        self._cparams = [cparam.instantiate() if isinstance(cparam, netutils.ghostvar) and cparam.instantiable else
                         cparam for cparam in self._cparams]
        self._state = [state.instantiate() if isinstance(state, netutils.ghostvar) and state.instantiable else
                       state for state in self._state]

    # Step method (same as feedforward)
    def step(self, inp):
        """Method to step through a unit in time. Useful only for recurrent layers."""
        return self.feedforward(inp=inp)

    # Feedforward method
    def feedforward(self, inp=None):
        """This method builds the actual theano graph linking input (self.x) to output (self.y)."""
        if inp is None:
            inp = self.x
        return inp

    # Decoder feedforward method
    def decoderfeedforward(self, inp=None):
        """This method builds the theano graph linking output (self.y) to reconstruction (self.xr)."""
        if inp is None:
            inp = self.y
        return inp

    # Apply parameters
    def applyparams(self, params=None, cparams=None):
        """This method applies numerical (or theano shared) parameters to the layer."""
        # Generic method for applying parameters
        if params is not None:
            # Convert to numeric (in case params is symbolic)
            params = netutils.sym2num(params)
            # Loop over all params, and set values
            for param, value in zip(self.params, params):
                param.set_value(value)

        if cparams is not None:
            # Convert to numeric
            cparams = netutils.sym2num(cparams)
            # Loop over all cparams and set values
            for cparam, value in zip(self.cparams, cparams):
                cparam.set_value(value)

    # Apply states
    def applystates(self, states=None):
        """This method applies numerical (or theano shared) states to the layer."""
        if states is not None:
            # Convert to numeric (in case params is symbolic)
            states = netutils.sym2num(states)
            # Loop over all params, and set values
            for state, value in zip(self.state, states):
                state.set_value(value)

    # Method to activate encoder or decoder
    def activate(self, what='all'):
        """Use this method to activate the encoder and/or decoder."""
        if what == 'enc' or what == 'all':
            self.encoderactive = True

        if what == 'dec' or what == 'all':
            self.decoderactive = True

    # Method to deactivate encoder or decoder
    def deactivate(self, what='all'):
        """Use this method to deactivate the encoder and/or decoder."""
        if what == 'enc' or what == 'all':
            self.encoderactive = False

        if what == 'dec' or what == 'all':
            self.decoderactive = False

    # Method for infering output shapes
    def inferoutshape(self, inpshape=None, checkinput=False):
        """Infer the output shape given an input shape. Required for automatic shape inference."""
        if inpshape is None:
            inpshape = self.inpshape
        return inpshape

    # Method to infer state shape
    def inferstateshape(self, inpshape=None):
        # Check if layer is stateful
        if self.isstateful:
            raise NotImplementedError("State shape inference not defined yet.")
        else:
            raise NotImplementedError("Layer is stateless.")

    def __add__(self, other):
        """Stack layers to build a network."""
        # Make sure the number of inputs/outputs check out
        assert self.numout == other.numinp, "Cannot chain a component with {} output(s) " \
                                            "with one with {} input(s)".format(self.numout, other.numinp)

        if isinstance(other, layertrain):
            # Make a layertrain only if chain is linear (i.e. no branches)
            # other.numout = 1 for other a layertrain
            if self.numinp > 1:
                return layertrainyard([self, other])
            else:
                return layertrain([self] + other.train)
        elif isinstance(other, layer):
            # Make a layertrain only if chain is linear (i.e. no branches)
            if all([num == 1 for num in [self.numinp, self.numout, other.numinp, other.numout]]):
                return layertrain([self] + [other])
            else:
                return layertrainyard([self, other])
        elif isinstance(other, layertrainyard):
            return layertrainyard([self] + other.trainyard)
        else:
            raise TypeError('Unrecognized layer class.')

    def __mul__(self, other):
        if isinstance(other, layertrain):
            return layertrainyard([[self, other]])
        elif isinstance(other, layer):
            return layertrainyard([[self, other]])
        elif isinstance(other, layertrainyard):
            return layertrainyard([[self, other]])
        else:
            raise TypeError('Unrecognized layer class.')

    def __div__(self, other):
        raise NotImplementedError("Div method not implemented...")

    def __pow__(self, power, modulo=None):
        raise NotImplementedError("Pow method not implemented...")

    def __repr__(self):
        # Layer spec string
        layerspec = "[{}]".format(self.layerinfo) if self.layerinfo is not None else ""

        desc = "--{}>>> {}{} >>>{}--".format(self.inpshape,
                                             self.__class__.__name__ + "({})".format(str(id(self))),
                                             layerspec,
                                             self.outshape)
        return desc


class recurrentchain(layer):
    """ Class to handle arbitrary recurrent chains. Wrapped as a layer for use in layertrains. """

    def __init__(self, chain, circuit=None, persistentmemory=False, returnsequence=True, truncatebptt=-1,
                 opencircuit=False, inpshape=None):
        """
        :type chain: list of layer
        :param chain: Layers to apply at constant time.
        :type circuit: numpy.ndarray
        :param circuit: Connectivity circuits. A 3D array with
                        circuit[0, ...] = data injection circuit,
                        circuit[1, ...] = time transfer circuit,
                        circuit[2, ...] = equal time circuit, and
                        circuit[3, ...] = data ejection circuit.
        :type persistentmemory: bool
        :param persistentmemory: Whether to have the cell state persist over function calls.

        :type returnsequence: bool
        :param returnsequence: Whether to return a sequence as output.

        :type truncatebptt: int
        :param truncatebptt: How many timesteps to truncate backprop through time (bptt) after.

        :type opencircuit: bool
        :param opencircuit: Whether to open the feedback circuit (for training). When set to true, y[t-1] is provided
                            as an input to the (recurrent) step function, where y[t] is the network output.

        :type inpshape: list or tuple
        :param inpshape: Input shape
        """

        # Init superclass
        super(recurrentchain, self).__init__()

        # Check input
        # chain = [chain] if not isinstance(chain, list) else chain
        assert all([isinstance(ring, (layer, layertrain)) for ring in chain]), "Chain must contain layers."
        assert isinstance(circuit, np.ndarray) or circuit is None, "Circuit must be a nd numpy array."

        # Much meta, such wow
        self._opencircuit = None
        self._chain = chain

        self._params = [param for ring in chain for param in ring.params]
        self._cparams = [cparam for ring in chain for cparam in ring.cparams]

        # Meta
        self.allowsequences = True
        self.issequence = True
        self.returnsequence = returnsequence
        self.encoderactive = True
        self.decoderactive = False
        self.dim = 2
        self.inpdim = 5
        self.truncatebptt = truncatebptt

        # Check if persistent memory is possible and requested
        self.persistentmemory = persistentmemory
        # self.persistentmemory = persistentmemory and all([ring.persistentmemory
        #                                                   if hasattr(ring, "persistentmemory")
        #                                                   else all([shp is not None for shp in ring.outshape])
        #                                                   for ring in self.chain])

        if not self.persistentmemory and persistentmemory:
            warn("Failed to initialize persistent memory.")

        if circuit is None:
            self._circuit = self.circuitgenerator()
        else:
            self._circuit = circuit

        # Shape inference
        if inpshape is None:
            self.inpshape = self.chain[0].inpshape
        else:
            assert len(inpshape) == self.inpdim, "Input must be {}D tensors.".format(self.inpdim)
            self.inpshape = inpshape

        # At present, recurrentchain can work with only one external input. This limitation is somewhat artificial,
        # considering that theano scan can handle multiple sequences, but required for recurrentchain to function as a
        # layer in layertrain, which was built with only one input (per layer) in mind.
        # UPDATE: This restriction has been relaxed with layertrainyard
        # Make sure there's only one incoming data injection connection
        # assert np.count_nonzero(self.circuit[0, ...]) == 1, "Recurrent chain supports only one input at present."

        # Compute the number of inputs
        self.numinp = np.count_nonzero(self.circuit[0, ...])
        # Compute the number of outputs
        self.numout = np.count_nonzero(self.circuit[3, ...])

        # Recurrence number equals the number of update slots and/or the number of outputs from step i.e. the number of
        # time transfer connections
        self.recurrencenumber = np.count_nonzero(self.circuit[1, ...])

        # Build state variable
        self.state, self._outnodeindex = self.buildstate()

        # Containers for input
        self.x = pyk.delist([T.tensor('floatX', [False, ] * self.inpdim, name="x{}:".format(inpnum) + str(id(self)))
                             for inpnum in range(self.numinp)])
        self.y = pyk.delist([T.tensor('floatX', [False, ] * self.inpdim, name="y{}:".format(outnum) + str(id(self)))
                             for outnum in range(self.numout)])

        # Open or close circuit
        if opencircuit:
            self.opencircuit = opencircuit
        else:
            self._opencircuit = False

    @property
    def circuit(self):
        return self._circuit

    @circuit.setter
    def circuit(self, value):
        # TODO: Assertions
        # Set
        self._circuit = value
        # FIXME: Provision for recurrencenumber = 0.5
        # Re-evaluate recurrence number and state
        self.recurrencenumber = np.count_nonzero(self.circuit[1, ...])
        # Rebuild state
        self.state, self._outnodeindex = self.buildstate(circuit=value)

    @property
    def chain(self):
        return self._chain

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        # Read numerical data from symbolic variables (shared variable or allocated tensor variable)
        value = netutils.sym2num(value)
        # Apply
        self.applyparams(params=value)

    @property
    def cparams(self):
        return self._cparams

    @cparams.setter
    def cparams(self, value):
        # Read numerical data from symbolic variables
        value = netutils.sym2num(value)
        # Apply
        self.applyparams(cparams=value)

    @property
    def opencircuit(self):
        return self._opencircuit

    @opencircuit.setter
    def opencircuit(self, value):
        value = bool(value)
        samevalue = self._opencircuit == value
        self._opencircuit = value

        if not samevalue:
            if value:
                # Circuit open. Add extra layer inputs
                self.numinp += len(self._outnodeindex)
                self.x = pyk.delist([T.tensor('floatX', [False, ] * self.inpdim,
                                              name="x{}:".format(inpnum) + str(id(self)))
                                     for inpnum in range(self.numinp)])
                # Change input shapes (override shape lock, since the output shape does not change)
                self._inpshape = pyk.list2listoflists(self.inpshape) + pyk.list2listoflists(self.outshape)

            else:
                # Circuit closed. Remove extra layer inputs
                self.numinp -= len(self._outnodeindex)
                self.x = pyk.delist([T.tensor('floatX', [False, ] * self.inpdim,
                                              name="x{}:".format(inpnum) + str(id(self)))
                                     for inpnum in self.numinp])
                # Change input shapes (override shape lock)
                self._inpshape = pyk.delist(self.inpshape[0:self.numinp])

    def step(self, *args):
        args = list(args)
        # Args is of the form: (inputarg1, ... inputargL, updateslot1, ... updateslotM), i.e. with L inputs and M update
        # slots.
        # Compute the number of input sequences (nis) and update slots (nus) this function expects.
        nis = np.count_nonzero(self.circuit[0, ...])
        nus = len(np.unique(np.array(np.where(self.circuit[1, ...] > 0)).T[:, 1]))

        # Before proceeding, check if the circuit is open. If it is, one (or more) of the inputs actually belong in
        # their respective update slot.
        if self.opencircuit:
            # self.outnode is a list of indices of model outputs in the output tuple of the step function. Not so
            # incidentally, this corresponds to indices in the list of update slots.
            # Compute the number of extraneous input arguments
            numextrargs = len(args) - (nis + nus)
            # List of extra inputs
            extrainps = args[nis:(nis + numextrargs)]
            # Make sure the number of extra inps is consistent with the list of output node indices
            assert len(extrainps) == len(pyk.obj2list(self._outnodeindex)), "Number of extra inputs must equal the " \
                                                                            "number of output nodes."
            # List of update slots
            us_ = args[-nus:]
            # List of input slots
            is_ = args[0:nis]
            # Loop over output node indices and assign to update slots
            for oni, eii in zip(pyk.obj2list(self._outnodeindex), range(len(extrainps))):
                us_[oni] = extrainps[eii]
            # Make a new args list
            args = is_ + us_

        # Seperate the inputs from the updates using the given circuit
        assert nis + nus == len(args), "Number of function arguments ({}) must equal the number of input sequences " \
                                       "({}) plus the number of update slots ({}).".format(len(args), nis, nus)
        # Separate inputs and update slots in two separate lists
        arginps = args[0:nis]
        argus = args[nis:]

        # Execution sequence: say chain = Ring 1 --> Ring 2 --> ... --> Ring 3, where:
        #   Ring 1 = LSTM1
        #   Ring 2 = CNN2
        #   Ring 3 = LSTM3
        # with hidden layer shared by all rings.

        # Circuit node list:
        #
        #             LSTM1                CNN2                   LSTM3
        #  0_________________________5  5___________7  7________________________12  <<< Chain Partition (chainpartition)
        # [input, hin, Cin, hout, Cout, input, output, input, hin, Cin, hout, Cout]

        # Number of input sequences: nis = sum(data injection circuit)
        # Number of update slots: nus = sum(time transfer circuit)

        # In this particular scenario, the update slots are:
        #   LSTM1.Cout(t) --> LSTM1.Cin(t + 1),
        #   LSTM3.hout(t) --> LSTM1.hin(t + 1),
        #   LSTM3.Cout(t) --> LSTM3.Cin(t + 1).
        # The dictionary cusmap maps index of the update variables in node list to corresponding index in the list of
        # update slots (argus). In this case: cusmap = {4: 0, 10: 1, 11: 2}.

        # To start, parse keys for cusmap from the given time transfer circuit
        # Fetch circuit
        ttc = self.circuit[1, ...]
        # Find connections
        ttcc = np.array(np.where(ttc > 0)).T
        # Find unique connection sources (ttc.shape = (numtargets, numsources))
        ttccs = np.unique(ttcc[:, 1])
        # Check if the number of sources equals the number of update slots
        assert ttccs.size == len(argus), "Number of unique time transfer circuit sources must equal the number of " \
                                         "update slots."
        # Save to cusmap (ni: node index, usi: update slot index)
        cusmap = {ni: usi for ni, usi in zip(ttccs, range(ttccs.size))}

        # Do just that for for the input sequences arginps (cinmap)
        # Fetch circuit
        dic = self.circuit[0, ...]
        # Find connections
        dicc = np.array(np.where(dic > 0)).T
        # Find unique connection sources
        diccs = np.unique(dicc[:, 1])
        # Check if the number of sources equals the number of input sequences
        assert diccs.size == len(arginps), "Number of unique input sequence sources must equal the number of " \
                                           "input sequences."
        # Save to cinmap
        cinmap = {ni: isi for ni, isi in zip(diccs, range(diccs.size))}

        # Generate chain partition
        chainpartition = []
        cursorstop = 0
        for ring in self.chain:
            chainpartition.append(slice(int(cursorstop), int(cursorstop + 2 * ring.recurrencenumber + 1)))
            cursorstop += int(2 * ring.recurrencenumber + 1)

        # Log number of nodes and instantiate a list of nodes
        numnodes = cursorstop
        # Make sure the number of nodes corresponds to that in circuit
        assert self.circuit.shape[1] == self.circuit.shape[2] == numnodes, "Number of nodes in the circuit graph " \
                                                                           "does not equal that required by the " \
                                                                           "given chain, check circuit."
        nodelist = [None for _ in range(numnodes)]
        # Cursors on node list
        nodelistcursor = 0

        # Loop over rings
        for ring, partition in zip(self.chain, chainpartition):
            # Init ring inputs
            ringinp = []
            # Compute the number of ring inputs = 1 + recurrence number
            numringinp = int(1 + ring.recurrencenumber)

            # Loop over ring inputs and check all circuits for incoming connections. If multiple incoming connections
            # are found, concatenate all before feeding to the layer's step function.
            for ringinpnum in range(numringinp):
                inp = []
                # Check all circuits (there are 3) for incoming connections, one by one
                # ---Data Injection Circuit---
                # Fetch subcircuit
                sdic = self.circuit[0, partition, :]
                # Detect any incoming input sequences and store to an array. For more commentary,
                # see corresponding line in time transfer circuit
                conns = np.array(np.where(sdic > 0)).T
                if conns[conns[:, 0] == ringinpnum].shape[0] != 0:
                    # Loop over incoming connection sources
                    for srcnum in conns[conns[:, 0] == ringinpnum][:, 1]:
                        # Look up corresponding index in the list of arguments and append to inp
                        inp.append(arginps[cinmap[srcnum]])
                else:
                    pass

                # ---Time Transfer Circuit---
                sttc = self.circuit[1, partition, :]
                # Find incoming connections and store to an array. Expected: conns.shape = (n, 2) where n is the
                # number of incoming connections. Conns is already sorted the way it should be.
                conns = np.array(np.where(sttc > 0)).T
                # Check whether there are any incoming connections in the current channel.
                if conns[conns[:, 0] == ringinpnum].shape[0] != 0:
                    # conns[conns[:, 0] == ringinpnum] gives all connections targeted at the current ring input.
                    # Loop over all sources and append to the input list inp
                    for srcnum in conns[conns[:, 0] == ringinpnum][:, 1]:
                        # Locate srcnum in cusmap and append. Remember that cusmap maps node index (srcnum) to argument
                        # index.
                        inp.append(argus[cusmap[srcnum]])
                else:
                    pass

                # ---Equal Time Circuit---A
                setc = self.circuit[2, partition, :]
                # Find incoming connections
                conns = np.array(np.where(setc > 0)).T
                # Check whether there are any incoming connections:
                if conns[conns[:, 0] == ringinpnum].shape[0] != 0:
                    # Loop over all sources and append to input list inp
                    for srcnum in conns[conns[:, 0] == ringinpnum][:, 1]:
                        # Fetch connection from nodelist and append to inp
                        inp.append(nodelist[srcnum])
                else:
                    pass

                # Concatenate all inputs along the channel axis
                if len(inp) > 1:
                    inp = T.concatenate(inp, axis=1)
                elif len(inp) == 1:
                    inp = inp[0]
                else:
                    raise LookupError("No incoming connections detected, check circuit.")

                # Append to ring inputs
                ringinp.append(inp)

            # Step through the ring
            ringout = pyk.obj2list(ring.step(*ringinp))
            ringnodes = ringinp + ringout
            assert len(ringnodes) == 2 * ring.recurrencenumber + 1, "Ring's step function did not return the " \
                                                                    "expected number of outputs = ring.recurrencenumber"

            # Populate nodelist
            nodelist[partition] = ringnodes

        # Make sure no nodes are left unassigned
        assert all([node is not None for node in nodelist]), "Nodelist not completely processed."

        # Prepare step function output.
        argout = [nodelist[ni] for ni in cusmap.iterkeys()]

        # Return
        return tuple(argout)

    def feedforward(self, inp=None):
        # Parse input
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        # Inp may be a list of multiple inputs
        # Inp is 5D sequential: (batchsize, T, fmapsin, y, x). Bring time axis (T) to front for scan:
        inp = [inpt.dimshuffle(1, 0, 2, 3, 4) for inpt in pyk.obj2list(inp)]

        # Initialize initial state. If memory is persistent, self.state would have shared variables only and no ghost
        # variables will be instantiated. Variable shapes will be instantiated according to the first input
        # FIXME: Variable instantiation must be circuit aware!
        initstate = [var.instantiate(inp[0].shape[1:]) if isinstance(var, netutils.ghostvar) else var
                     for var in self.state]

        # Scan
        finalstate, log = th.scan(self.step, sequences=inp, outputs_info=[{"initial": state} for state in initstate],
                                  truncate_gradient=self.truncatebptt)

        # Place parameter update requests if requested. Remember that all variables in initstate are
        # shared if memory is persistent. Look at the corresponding line in LSTM's feedforward method to understand the
        # use of dictionaries.
        if self.persistentmemory:
            updict = dict(self.updaterequests)
            updict.update({istate: fstate[-1] for istate, fstate in zip(initstate, finalstate)})
            self.updaterequests = updict.items()

        # Prepare output
        # Feedforward method supports only one output at the moment
        assert len(self._outnodeindex) == 1 if isinstance(self._outnodeindex, list) else True, \
            "Only one output node supported at present."
        # Fetch output node index and convert to integer
        oni = int(self._outnodeindex[0] if isinstance(self._outnodeindex, list) else self._outnodeindex)

        if not self.returnsequence:
            # Read output from final state and return as an image (numimg, numch, y, x)
            out = finalstate[oni][-1]
        else:
            # Read output from final state and return as a sequence (numimg, T, numch, y, x)
            out = finalstate[oni].dimshuffle(1, 0, 2, 3, 4)

        # Reshuffle dimensions to put the time axis where it belongs
        self.y = out

        # Return
        return self.y

    # Method to build the state of the recurrentchain from circuit
    def buildstate(self, circuit=None):
        # Recall from self.step():
        # Circuit node list:
        #
        #             LSTM1                CNN2                   LSTM3
        #  0_________________________5  5___________7  7________________________12  <<< Chain Partition (chainpartition)
        # [input, hin, Cin, hout, Cout, input, output, input, hin, Cin, hout, Cout] <<< Circuit node list
        # -----------n = 2------------  ---n = 0.5---  -----------n = 2-----------  <<< Recurrence numbers
        # Parse circuit
        if circuit is None:
            circuit = self.circuit

        # Fetch time transfer circuit
        ttc = circuit[1, ...]
        # Find connections
        ttcc = np.array(np.where(ttc > 0)).T
        # Find unique connection sources (ttc.shape = (numtargets, numsources)). This should be a list of indices in the
        # node list. In the above example with
        #   LSTM1.Cout(t) ---> LSTM1.Cin(t + 1),
        #   LSTM3.hout(t) ---> LSTM3.hin(t + 1),
        #   LSTM3.Cout(t) ---> LSTM3.Cin(t + 1)
        # one would expect: ttccs = [4, 10, 11]
        ttccs = np.unique(ttcc[:, 1])
        # Find the corresponding variables in the corresponding ring. To do that, remember:
        #   LSTM1.step(input, hin, Cin) = (hout, Cout)
        # generalizes to:
        #   RECUNIT.step(input, hin, ...) = (hout, ...)
        # where "..." could be arbitrary in size, but hout remains the primary unit output.
        # To start sorting out this mess:
        #   [v] Get a list of slices corresponding to the node indicies of nodes belonging to a ring. (1)
        #   [v] Find the parameters within the ring, or make new parameters (as ghost variables) on the go. (2)
        #   [v] Write to recurrentchain state. (3)

        # (1)
        # Chainpartition with the above model: [slice(0, 5), slice(5, 7), slice(7, 12)]
        chainpartition = []
        cursorstop = 0
        for ring in self.chain:
            chainpartition.append(slice(int(cursorstop), int(cursorstop + 2 * ring.recurrencenumber + 1)))
            cursorstop += int(2 * ring.recurrencenumber + 1)

        # Figure out which state variables belong to which ring. Sourceinring from the above example: [0, 2, 2]
        sourceinring = [[s in range(ringpartition.start, ringpartition.stop)
                         for ringpartition in chainpartition].index(True) for s in ttccs]

        # (2)
        # Discover state variables in rings. If memory is persistent, state variables are certainly shared variables.
        statevars = [self.chain[ri].state[ni - chainpartition[ri].start - self.chain[ri].recurrencenumber - 1]
                     if self.chain[ri].recurrencenumber != 0.5
                     else netutils.ghostvar(shape=self.chain[ri].outshape,
                                            shared=self.persistentmemory,
                                            value=lambda ishp: np.random.uniform(size=ishp),
                                            name="recchain-node{}-ring{}:{}".format(ni, ri, str(id(self))))
                     for ni, ri in zip(ttccs, sourceinring)]

        # (3)
        # Instantiate ghost variables. If memory is persistent, statevars should contain only shared variables.
        # Otherwise, it should contain ghost variables.
        statevars = [var.instantiate() if isinstance(var, netutils.ghostvar) and var.instantiable
                     else var for var in statevars]

        # Parse output node index. Look for the node index of the output source in data ejection circuit
        # Fetch data ejection circuit
        dec = circuit[3, ...]

        # Find connections. decc.shape = (numtargets, numsources)
        decc = np.array(np.where(dec > 0)).T

        # Find unique connection sources.
        # Remember:
        #   > The output must also be a source in time transfer circuit, i.e. a recurrent/state variable
        #   > Recurrentchain supports only one output when working with layertrain. What's done below is however more
        #     general.
        deccs = np.unique(decc[:, 1])

        # Find index in ttccs of elements in deccs (read: deccs in ttccs). Index in ttccs corresponds 1-to-1 to index in
        # statevars.
        deccsinttccs = [list(ttccs).index(x) for x in deccs]

        # Return
        return statevars, deccsinttccs

    # Method to generate circuit
    def circuitgenerator(self, debug=True):

        if debug:
            raise NotImplemented("Circuit generator is yet to be correctly implemented. Please assign to circuit manually.")

        # Generate chain partition
        chainpartition = []
        cursorstop = 0
        for ring in self.chain:
            chainpartition.append(slice(int(cursorstop), int(cursorstop + 2 * ring.recurrencenumber + 1)))
            cursorstop += int(2 * ring.recurrencenumber + 1)

        # Cursorstop now gives the size of the circuit matrix - init a zero matrix
        circuit = np.zeros(shape=(4, cursorstop, cursorstop))

        # Variables to cache the emit and receive nodes in equal time circuit
        emitnode = None
        # Iterate over chain and build circuit.
        for ringnum, ring in enumerate(self.chain):
            # Data injection circuit: Set to that of the first ring
            if ringnum == 0:
                circuit[0, chainpartition[ringnum], chainpartition[ringnum]] = ring.circuit[0, ...]

            # Time transfer circuit
            circuit[1, chainpartition[ringnum], chainpartition[ringnum]] = ring.circuit[1, ...]

            # Equal time circuit
            # Determine nodes of incoming and outgoing connections.
            # If ring is first in chain, fetch emitnode and ignore receive node (incoming data is provided via time
            # transfer circut and/or data injection circuit).
            # FIXME DRY this code-snippet
            if ringnum == 0:
                # Fetch emit node.
                # Find all nodes in equal time circuit. The following expression yields a n-by-2 array where the rows
                # are 1-by-2 arrays of nodes, i.e. (targetindex, sourceindex).
                etcnodes = np.array(np.where(ring.circuit[2, ...] == -1)).T
                # Make sure etc circuit is not empty
                assert len(etcnodes) is not 0, "No equal time circuit nodes found."
                # If multiple outgoing connections are found, pick the first and discard the rest
                emitnode = etcnodes[0, 1] + chainpartition[ringnum].start

            elif ringnum == len(self.chain):
                # Find all nodes in equal time circuit. The following expression yields a n-by-2 array where the rows
                # are 1-by-2 arrays of nodes, i.e. (targetindex, sourceindex).
                etcnodes = np.array(np.where(ring.circuit[2, ...] == -1)).T
                assert len(etcnodes) is not 0, "No equal time circuit nodes found."
                # Find the receiving node
                receivenode = etcnodes[0, 0] + chainpartition[ringnum].start
                # Set corresponding node in equal time circuit of the recurrent block
                circuit[receivenode, emitnode] = 1

            else:
                # This is the case where a ring is neither first nor last in chain.
                # Find all nodes in equal time circuit
                etcnodes = np.array(np.where(ring.circuit[2, ...] == -1)).T
                assert len(etcnodes) is not 0, "No equal time circuit nodes found."
                # Find receiving node in the current ring
                receivenode = etcnodes[0, 0] + chainpartition[ringnum].start
                # Set corresponding node in equal time circuit of the recurrent block
                circuit[receivenode, emitnode] = 1
                # Find the new emit node, i.e. that of the current chain
                emitnode = etcnodes[0, 1] + chainpartition[ringnum].start

            # Data Ejection Circuit
            # Copy data ejection circuit of the last ring in chain to that of the recurrent chain
            if ringnum == len(self.chain):
                circuit[3, chainpartition[ringnum], chainpartition[ringnum]] = ring.circuit[3, ...]

        # return
        return circuit

    # Method to apply parameters
    def applyparams(self, params=None, cparams=None):

        # Params is a list of parameters of all rings. Parse how many parameters belong to which ring
        lenlist = [len(ring.params) for ring in self.chain]

        # Declare cursors
        cursorstart = cursorstop = 0

        # Loop over rings
        for ringnum, ring in enumerate(self.chain):

            # Skip if there are no parameters to be applied
            if lenlist[ringnum] == 0:
                continue

            # Figure out where to place the stop cursor
            cursorstop = cursorstart + lenlist[ringnum]

            # Fetch parameters and cparams to apply
            if params is not None:
                ringparams = params[cursorstart:cursorstop]
            else:
                ringparams = None

            if cparams is not None:
                ringcparams = cparams[cursorstart:cursorstop]
            else:
                ringcparams = None

            # Apply parameters
            ring.applyparams(params=ringparams, cparams=ringcparams)

            # Place start cursor for the next iteration
            cursorstart = cursorstop

    # FIXME: Shape inference must be circuit aware
    # Shape inference
    def inferoutshape(self, inpshape=None, checkinput=False):
        # Parse inpshape
        if inpshape is None:
            inpshape = self.inpshape

        # Warning about debug mode
        if not self.shapelock:
            warn("Shape inference is not correctly implemented for this layer. Ignore this message if inpshape was set "
                 "manually. Otherwise, set inpshape manually and arm shapelock.")

        # Shape is a buffer variable
        shape = inpshape
        for ring in self.chain:
            # Inpshape is expected of shape (nb, T, nc, r, c), but the T axis is handled by the recurrent chain itself.
            # The step functions of rings in chain have no business worrying about the T axis; this is taken care of
            # by their respective feedforward() methods or alternatively, this recurrent chain.
            # Strip T dimension if it exists
            shape = ((shape[0:1] + shape[2:]) if len(shape) == 5 else shape)
            # In an ideal world, that would be sufficient. But some layers are allowed to complain if their input is
            # not 5D sequential. If that's the case, add a singleton T dimension.
            shape = shape[0:1] + ([1] if ring.inpdim == 5 else []) + shape[1:]
            # ring.inpshape has a setter that sets outshape
            ring.inpshape = shape
            shape = ring.outshape
        # Make sure output shape is 5D sequential
        shape = (shape[0:1] + inpshape[1:2] + shape[1:]) if len(shape) == 4 else shape
        return shape


class convlayer(layer):
    """ Convolutional Layer """

    # Constructor
    def __init__(self, fmapsin, fmapsout, kersize, stride=None, padding=None, dilation=None, activation=netools.linear(),
                 alpha=None, makedecoder=False, zerobias=False, tiedbiases=True, convmode='same', allowsequences=True,
                 inpshape=None, W=None, b=None, bp=None, Wc=None, bc=None, bpc=None, Wgc=None, bgc=None, bpgc=None,
                 allowgradmask=False):

        """
        :type fmapsin: int
        :param fmapsin: Number of input feature maps

        :type fmapsout: int
        :param fmapsout: Number of output feature maps

        :type kersize: tuple or list
        :param kersize: Size of the convolution kernel (y, x, z); A 2-tuple (3-tuple) initializes a 2D (3D) conv. layer

        :type stride: tuple or list
        :param stride: Convolution strides. Must be a 2-tuple (3-tuple) for a 2D (3D) conv. layer.
                       Defaults to (1, ..., 1).

        :type dilation: tuple or list
        :param dilation: Dilation for dilated convolutions.

        :type activation: dict or callable
        :param activation: Transfer function of the layer.
                           Can also be a dict with keys ("function", "extrargs", "train") where:
                                function: elementwise theano function
                                trainables: extra parameter variables (for PReLU, for instance)
                                ctrainables: extra parameter connectivity variables

        :type alpha: float
        :param alpha: Initialization gain (W ~ alpha * N(0, 1))

        :type makedecoder: bool
        :param makedecoder: Boolean switch for initializing decoder biases

        :type zerobias: bool
        :param zerobias: Whether not to use bias. True => no bias used (also not included in params).

        :type tiedbiases: bool
        :param tiedbiases: Decoder bias = - Encoder bias when set to True

        :type convmode: str
        :param convmode: Convolution mode. Possible values: 'same' (default), 'full' or 'valid'

        :type allowsequences: bool
        :param allowsequences: Whether to process 3D data as sequences. When set to true and the kernel looks something
                               like [ky, kx, 1] (i.e. kernel[2] = 1), the 3D data is processed by 2D operations
                               (2D convolution).

        :type inpshape: tuple or list
        :param inpshape: Input shapes to expect. Used for optimizing convolutions and recurrent chains.

        :type W: theano tensor of size (fmapsout, kersize[2], fmapsin, kersize[0], kersize[1]) in 3D,
                                       (fmapsout, fmapsin, kersize[0], kersize[1]) in 2D
        :param W: Preset weight tensor of the layer (use for tying weights)

        :type b: theano vector of size (fmapsout,)
        :param b: Preset bias vector of the layer (use for tying weights)

        :type bp: theano vector of size (fmapsin,)
        :param bp: Preset bias vector of the associated decoder layer (use for tying weights)

        :type Wc: Floated boolean theano tensor of shape identical to that of W
        :param Wc: Connectivity mask of the weight tensor. For a tensor element set to zero, the corresponding element
                   in the weight tensor never gets updated. This can be exploited to have and train two parallel
                   'sublayers' without one interfering with the other.


        :type bc: Floated boolean theano tensor of shape identical to that of b
        :param bc: Connectivity mask of the bias vector. For more documentation see: Wc.

        :type bpc: Floated boolean theano tensor of shape identical to that of bpc
        :param bpc: Connectivity mask of the decoder bias vector. For more documentation see: Wc.

        :type allowgradmask: bool
        :param allowgradmask: Whether to allow gradient masking. There's no reason not to as such, except when using a
                              recurrent layer, the gradient computation fails (known problem).

        :return: None
        """

        # Initialize super class
        super(convlayer, self).__init__()

        # Meta
        self.fmapsin = int(fmapsin)
        self.fmapsout = int(fmapsout)
        self.kersize = list(kersize)
        self.decoderactive = bool(makedecoder)
        self.encoderactive = True
        self.zerobias = bool(zerobias)
        self.tiedbiases = bool(tiedbiases)
        self.convmode = str(convmode)
        self.allowsequences = bool(allowsequences)
        self.allowgradmask = allowgradmask

        # Parse activation
        if isinstance(activation, dict):
            self.activation = activation["function"]
            self.extratrainables = activation["trainables"] if "trainables" in activation.keys() else []
            self.extractrainables = activation["ctrainables"] if "ctrainables" in activation.keys() else \
                [netutils.getshared(like=trainable, value=1.) for trainable in self.extratrainables]
        elif callable(activation):
            self.activation = activation
            self.extratrainables = []
            self.extractrainables = []
        else:
            self.activation = netools.linear()
            self.extratrainables = []
            self.extractrainables = []

        # Name extra trainables for convenience
        for trainablenum, trainable in enumerate(self.extratrainables):
            trainable.name = trainable.name + "-trainable{}:".format(trainablenum) + str(id(self)) \
                if trainable.name is not None else "trainable{}:".format(trainablenum) + str(id(self))
        for trainablenum, trainable in enumerate(self.extractrainables):
            trainable.name = trainable.name + "-ctrainable{}:".format(trainablenum) + str(id(self)) \
                if trainable.name is not None else "ctrainable{}:".format(trainablenum) + str(id(self))

        # Debug Paramsh
        # Encoder and Decoder Convolution Outputs
        self.eIW = None
        self.dIW = None
        # Encoder and Decoder Preactivations
        self.ePA = None
        self.dPA = None

        # Parse initialization alpha
        if alpha is None:
            self.alpha = 1.
            # self.alpha = np.sqrt(1. / (fmapsin * np.prod(kersize)))
        else:
            self.alpha = alpha

        # Parse network dimension (and whether the input a sequence)
        self.issequence = len(self.kersize) == 3 and self.kersize[2] == 1 and self.allowsequences
        self.dim = (2 if len(self.kersize) == 2 or (self.issequence and self.allowsequences) else 3)

        if self.dim == 2:
            self.inpdim = (4 if not self.issequence else 5)
        elif self.dim == 3:
            self.inpdim = 5
        else:
            raise NotImplementedError('Invalid layer dimension: {}. Supported: 2D and 3D.'.format(self.dim))

        # Parse convolution strides
        if stride is None:
            self.stride = [1, ] * (self.dim + (0 if not self.issequence else 1))
        else:
            if self.dim == 2:
                stride = [stride, ] * (self.dim + (0 if not self.issequence else 1)) if isinstance(stride, int) \
                    else stride
                assert len(stride) == len(self.kersize), "Stride and kersize must have the same length."
                self.stride = list(stride)
            else:
                warn("Convolution strides are presently not supported for 3D convolutions.")
                self.stride = [1, ] * (self.dim + (0 if not self.issequence else 1))

        # Parse dilation
        if dilation is None:
            self.dilation = [1, ] * (self.dim + (0 if not self.issequence else 1))
        else:
            if self.dim == 2:
                dilation = [dilation, ] * (self.dim + (0 if not self.issequence else 1)) if isinstance(dilation, int) \
                    else dilation
                assert len(dilation) == len(self.kersize), "Dilation and kersize must have the same length."
                assert self.stride == [1, 1], "Stride must be [1, 1] for dilated convolutions."
                self.dilation = list(dilation)
            else:
                warn("Dilated convolutions are presently not supported for 3D convolutions.")
                self.dilation = [1, ] * (self.dim + (0 if not self.issequence else 1))

        # Parse padding
        if padding is None:
            # Check if convolution is strided, convmode is 'same' but no padding is provided
            if not all([st == 1 for st in self.stride]) and self.convmode is 'same':
                # Try to infer padding for stride 2 convolutions with odd kersize
                if self.stride == [2, 2] and all([ks % 2 == 1 for ks in self.kersize]):
                    self.padding = [[(ks - 1)/2] * 2 for ks in self.kersize]
                    self.convmode = 'valid'
                else:
                    raise NotImplementedError("Padding could not be inferred for the strided convolution in the 'same' "
                                              "mode. Please provide manually. ")
            else:
                self.padding = [[0, 0], ] * {4: 2, 5: 3}[self.inpdim]
        else:
            assert len(padding) == {4: 2, 5: 3}[self.inpdim], "Padding must be a 3-tuple for 3D or 2D sequential data, " \
                                                              "2-tuple for 2D data."
            # Padding must be [[padleft, padright], ...]
            padding = [[padval, padval] if isinstance(padval, int) else padval[0:2] for padval in padding]
            self.padding = padding
            # Change convmode to valid with a warning
            if not all([st == 1 for st in self.stride]) and self.convmode is 'same':
                warn("Convlayer will apply 'valid' strided convolution to the padded input.")
                self.convmode = 'valid'

        # Initialize weights W and biases b:
        # W.shape = (fmapsout, kersize[2], fmapsin, kersize[0], kersize[1])     [3D]
        # W.shape = (fmapsout, fmapsin, kersize[0], kersize[1])                 [2D]
        # b.shape = (fmapsout,)

        # Weights, assoc. connectivity mask and gradient clips
        if W is None:
            # Fetch default init scheme (xavier)
            initscheme = netools.xavier
            if self.dim == 3:
                self.W = th.shared(
                    value=self.alpha * initscheme(shape=(fmapsout, kersize[2], fmapsin, kersize[0], kersize[1])),
                    name='convW:' + str(id(self)))
            else:
                self.W = th.shared(
                    value=self.alpha * initscheme(shape=(fmapsout, fmapsin, kersize[0], kersize[1])),
                    name='convW:' + str(id(self)))

        elif isinstance(W, str):
            if W in ["id", "identity"]:
                if self.dim == 2:
                    self.W = netools.idkernel([fmapsout, fmapsin, kersize[0], kersize[1]])
                else:
                    self.W = netools.idkernel([fmapsout, kersize[2], fmapsin, kersize[0], kersize[1]])
            else:
                # Parse init scheme
                initscheme = netutils.smartfunc(getattr(netools, W), ignorekwargssilently=True)
                # Parse kernel shape
                kershape = (fmapsout, kersize[2], fmapsin, kersize[0], kersize[1]) if self.dim == 3 else \
                    (fmapsout, fmapsin, kersize[0], kersize[1])
                # Initialize
                self.W = th.shared(value=self.alpha * initscheme(shape=kershape, dtype=th.config.floatX))

            self.W.name = 'convW:' + str(id(self))

        elif callable(W):
            # Parse init scheme
            initscheme = netutils.smartfunc(W, ignorekwargssilently=True)
            # Parse kernel shape
            kershape = (fmapsout, kersize[2], fmapsin, kersize[0], kersize[1]) if self.dim == 3 else \
                (fmapsout, fmapsin, kersize[0], kersize[1])
            # Initialize
            self.W = th.shared(value=self.alpha * initscheme(shape=kershape, dtype=th.config.floatX),
                               name='convW:' + str(id(self)))

        else:
            # W must be a shared variable.
            assert netutils.isshared(W), "W must be a shared variable."
            # W must have the right shape!
            Wshape = W.get_value().shape
            kershape = (fmapsout, kersize[2], fmapsin, kersize[0], kersize[1]) if self.dim == 3 else \
                (fmapsout, fmapsin, kersize[0], kersize[1])
            assert Wshape == kershape, "W is of the wrong shape. Expected shape: {}".format(kershape)

            self.W = W
            self.W.name = 'convW:' + str(id(self))

        # Conn. mask
        self.Wc = netutils.getshared(value=(1. if Wc is None else Wc), like=self.W, name='convWc:' + str(id(self)))

        # Gradient clips
        if Wgc is None:
            self.Wgc = [-np.inf, np.inf]
        else:
            assert isinstance(Wgc, (list, np.ndarray)) and len(Wgc) == 2, "Weight filter gradient clips (Wgc) must " \
                                                                          "be a list with two elements."
            self.Wgc = Wgc

        # Biases and conn. mask
        self.b = netutils.getshared(value=(0. if b is None else b), shape=(fmapsout,), name='convb:' + str(id(self)))

        # Conn. mask
        if bc is None and not self.zerobias:
            self.bc = netutils.getshared(value=1., like=self.b, name='convbc:' + str(id(self)))
        elif self.zerobias and b is None:
            self.bc = netutils.getshared(value=0., like=self.b, name='convbc:' + str(id(self)))
        else:
            self.bc = netutils.getshared(value=bc, like=self.b, name='convbc:' + str(id(self)))

        # Gradient clips
        if not bgc and not self.zerobias:
            self.bgc = [-np.inf, np.inf]
        elif self.zerobias and bgc is None:
            self.bgc = [0, 0]
        else:
            assert isinstance(bgc, (list, np.ndarray)) and len(bgc) == 2, "Bias gradient clips (bgc) must " \
                                                                          "be a list with two elements."
            self.bgc = bgc

        # Fold Parameters
        self._params = [self.W] + ([self.b] if not self.zerobias else []) + self.extratrainables

        self._cparams = [self.Wc] + ([self.bc] if not self.zerobias else []) + self.extractrainables

        # Parse input shape
        if inpshape is None:
            self.inpshape = [None, ] * self.inpdim
        else:
            assert len(inpshape) == self.inpdim, "Input shape must have exactly as many elements as input dimension."
            self.inpshape = inpshape

        # Parse output shape
        self.outshape = self.inferoutshape()

        # Container for input (see feedforward() for input shapes) and output
        # Input
        self.x = T.tensor('floatX', [False, ] * self.inpdim, name='x:' + str(id(self)))
        # Output
        self.y = T.tensor('floatX', [False, ] * self.outdim, name='y:' + str(id(self)))

        self.layerinfo = "{}-in >> {}-out w/ {} kernel".format(fmapsin, fmapsout, kersize)

    # Params and cparams property definitions
    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        # Piggy back on applyparams for back-compatibility
        self.applyparams(params=value)

    @property
    def cparams(self):
        return self._cparams

    @cparams.setter
    def cparams(self, value):
        self.applyparams(cparams=value)

    # Feed forward through the layer
    def feedforward(self, inp=None, activation=None):
        # Argument inp is expected of the form:
        #    inp.shape = (numimages, z, fmapsin, y, x)     [3D]
        #    inp.shape = (numimages, fmapsin, y, x)        [2D]
        # Setting reshape to True assumes that the 3D input is of the form (numimages, fmapsin, y, x, z)

        # Parse input
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        if not self.encoderactive:
            self.y = inp
            return inp

        if not activation:
            activation = self.activation

        # Check if gradient masking is required
        filtergradmask = self.Wc if self.allowgradmask else None
        biasgradmask = self.bc if self.allowgradmask else None

        # Get PreActivation
        PA = A.conv(inp, self.W, stride=self.stride, dilation=self.dilation, padding=self.padding, bias=self.b,
                    filtergradmask=filtergradmask, biasgradmask=biasgradmask, filtergradclips=self.Wgc,
                    biasgradclips=self.bgc, dim=self.dim, convmode=self.convmode, issequence=self.issequence)

        # Apply activation function
        self.y = activation(PA)
        # Return
        return self.y

    # Method to infer output shape
    def inferoutshape(self, inpshape=None, checkinput=True):
        # Parse
        if inpshape is None:
            inpshape = self.inpshape

        # Return input shape if encoder not active
        if not self.encoderactive:
            return inpshape

        # Check if input shape valid
        if checkinput:
            assert inpshape[(1 if self.inpdim == 4 else 2)] == self.fmapsin or \
                   inpshape[(1 if self.inpdim == 4 else 2)] is None, "Number of input channels must match that of " \
                                                                     "the layer."

        if self.inpdim == 4:
            # Compute y and x from input
            y, x = [((inpshape[sid] + (1 if self.convmode is 'full' else 0 if self.convmode is 'same' else -1) *
                    (self.kersize[kid] - 1) + (self.stride[kid] - 1) + sum(self.padding[kid])) // (self.stride[kid])
                     if inpshape[sid] is not None else None) for sid, kid in zip([2, 3], [0, 1])]
            # Fetch batchsize and number of output maps
            fmapsout = self.fmapsout
            batchsize = inpshape[0]

            return [batchsize, fmapsout, y, x]

        elif self.inpdim == 5:
            assert len(self.kersize) == 5 or self.issequence, "Layer must be 3D convolutional or sequential " \
                                                              "for 5D inputs."
            # Compute y, x and z from input
            y, x, z = [((inpshape[sid] + (1 if self.convmode is 'full' else 0 if self.convmode is 'same' else -1) *
                        (self.kersize[kid] - 1) + (self.stride[kid] - 1) + sum(self.padding[kid])) // (self.stride[kid])
                        if inpshape[sid] is not None else None) for sid, kid in zip([3, 4, 1], [0, 1, 2])]
            # Fetch batchsize and number of output maps
            fmapsout = self.fmapsout
            batchsize = inpshape[0]
            return [batchsize, z, fmapsout, y, x]

        pass



class poollayer(layer):
    """ General Max-pooling Layer """

    # Constructor
    def __init__(self, ds, stride=None, ignoreborder=True, padding=(0, 0), poolmode='max', switchmode=None,
                 switch=None, allowsequences=True, makedecoder=True, inpshape=None):
        """
        :type ds: 2- or 3-tuple for 2 or 3 dimensional network
        :param ds: tuple of downsampling ratios

        :type stride: tuple or list
        :param stride: Pooling strides. Defaults to ds (i.e. non-overlapping regions).

        :type padding: tuple or list
        :param padding: Input padding. Handled by Theano's max_pool_2D.

        :type ignoreborder: bool
        :param ignoreborder: Whether to ignore borders while pooling. Equivalent to ignore_border in Theano's
                             max_pool_2d.

        :type switchmode: str
        :param switchmode: Whether to use switches for unpooling. Layer must be 2 dimensional to use switches and the
                       possible options are:
                         None   :   No switches used
                        'hard'  :   Hard switches (1 for the max value, 0 for everything else in the window)
                        'soft'  :   Soft switches (proportion preserving switches: 1 for max, 0.5 for a value half that
                                    of max, etc.)

        :type switch: theano.tensor.var.TensorVariable
        :param switch: Optional input slot for a switch variable. Overrides switch variable computed in layer.

        :type allowsequences: bool
        :param allowsequences: Whether to allow sequences. If set to true and ds[2] = 1, 3D spatiotemporal data is
                               spatially pooled.

        :type inpshape: list or tuple
        :param inpshape: Expected input/output shape.

        """

        # Initialize super class
        super(poollayer, self).__init__()

        # Input check
        assert len(ds) == 2 or len(ds) == 3 or ds == 'global', "ds can only be a vector/list of length 2 or 3 or " \
                                                               "'global'."

        # Check if global pooling
        if ds == 'global':
            # Only 2D global pooling is supported right now
            ds = ['global', 'global']
            # TODO continue

        # Meta
        self.ds = ds
        self.poolmode = poolmode
        self.stride = self.ds if stride is None else stride
        self.ignoreborder = ignoreborder
        self.padding = padding
        self.decoderactive = makedecoder
        self.encoderactive = True
        self.switchmode = switchmode
        self.allowsequences = allowsequences

        # Define dummy parameter lists for compatibility with layertrain
        self.params = []
        self.cparams = []

        if padding == 'auto':
            # Decide padding based on stride
            if self.stride == [2, 2]:
                if all([dsp % 2 == 1 for dsp in self.ds]):
                    self.padding = [[(dsp - 1) / 2] * 2 for dsp in self.ds]
                else:
                    raise NotImplementedError("Poollayer cannot infer padding for window size "
                                              "{} and stride {}. Please provide padding manually.".format(self.ds,
                                                                                                          self.stride))
        elif isinstance(padding, (tuple, list)):
            self.padding = list(padding)
        else:
            raise NotImplementedError("Padding must be a tuple or a list or keyword 'auto'.")

        # Parse network dimension (and whether the input a sequence)
        self.issequence = len(self.ds) == 3 and self.ds[2] == 1 and self.allowsequences
        self.dim = (2 if len(self.ds) == 2 or (self.issequence and self.allowsequences) else 3)

        if self.dim == 2:
            self.inpdim = (4 if not self.issequence else 5)
        elif self.dim == 3:
            self.inpdim = 5
        else:
            raise NotImplementedError('Invalid layer dimension: {}. Supported: 2D or 3D.'.format(self.dim))

        # Switches
        # Check if switched unpooling possible
        assert not (switchmode is not None and self.dim == 3), "Switched unpooling implemented in 2D only."
        assert switchmode in [None, 'soft', 'hard'], "Implemented switch modes are 'soft' and 'hard'."

        # Check if switch variable provided
        if switch is not None:
            self.switch = switch
            self.switchmode = 'given'
        else:
            self.switch = T.tensor('floatX', [False, False, False, False], name='sw:' + str(id(self)))

        # Input and output shapes
        if inpshape is None:
            self.inpshape = [None, ] * self.inpdim
        else:
            assert len(inpshape) == self.inpdim, "Length of inpshape must match equal the number of input dimensions."
            self.inpshape = inpshape

        self.outshape = self.inferoutshape()

        # Containers (see feedforward() for input shapes)
        # Input
        self.x = T.tensor('floatX', [False, ] * self.inpdim, name='x:' + str(id(self)))
        # Output
        self.y = T.tensor('floatX', [False, ] * self.outdim, name='y:' + str(id(self)))

        self.layerinfo = "[{}-pool by {} kernel]".format(self.poolmode, self.ds)

    # Feed forward through the layer (pool)
    def feedforward(self, inp=None, reshape=False):
        # Argument inp is expected of the form:
        #    inp.shape = (numimages, z, fmapsin, y, x)     [3D]
        #    inp.shape = (numimages, fmapsin, y, x)        [2D]
        # Setting reshape to True assumes that the 3D input is of the form (numimages, fmapsin, y, x, z)

        # Parse input
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        # Check if encoder active
        if not self.encoderactive:
            self.y = inp
            return inp

        self.y = A.pool(inp=inp, ds=self.ds, stride=self.stride, padding=self.padding, poolmode=self.poolmode,
                        dim=self.dim, ignoreborder=self.ignoreborder, issequence=self.issequence)
        return self.y

    # Infer output shape
    def inferoutshape(self, inpshape=None, checkinput=False):
        # Parse
        if inpshape is None:
            inpshape = self.inpshape

        # Return input shape if encoder not active
        if not self.encoderactive:
            return inpshape

        # Process
        if self.inpdim == 4:
            if self.ignoreborder:
                y, x = [int(np.floor((inpshape[sid] + 2 * self.padding[kid] - self.ds[kid] + self.stride[kid]) /
                                     self.stride[kid])) if inpshape[sid] is not None else None
                        for sid, kid in zip([2, 3], [0, 1])]
            else:
                plen = [None, None]
                for sid, kid in zip([2, 3], [0, 1]):
                    if self.stride[kid] >= self.ds[kid]:
                        plen[kid] = int(np.floor((inpshape[sid] + self.stride[kid] - 1) / self.stride[kid])) \
                            if inpshape[sid] is not None else None
                    else:
                        plen[kid] = np.maximum(0, np.floor((inpshape[sid] - self.ds[kid] + self.stride[kid] - 1) /
                                                           self.stride[kid])) if inpshape[sid] is not None else None
                y, x = plen

            fmapsout = inpshape[1]
            batchsize = inpshape[0]

            return [batchsize, fmapsout, y, x]

        elif self.inpdim == 5:
            if self.ignoreborder:
                y, x, z = [int(np.floor((inpshape[sid] + 2 * self.padding[kid] - self.ds[kid] + self.stride[kid]) /
                                        self.stride[kid])) if inpshape[sid] is not None else None
                           for sid, kid in zip([3, 4, 1], [0, 1, 2])]
            else:
                plen = [None, None, None]
                for sid, kid in zip([3, 4, 1], [0, 1, 2]):
                    if self.stride[kid] >= self.ds[kid]:
                        plen[kid] = int(np.floor((inpshape[sid] + self.stride[kid] - 1) / self.stride[kid])) \
                            if inpshape[sid] is not None else None
                    else:
                        plen[kid] = np.maximum(0, np.floor((inpshape[sid] - self.ds[kid] + self.stride[kid] - 1) /
                                                           self.stride[kid])) if inpshape[sid] is not None else None
                y, x, z = plen

            fmapsout = inpshape[2]
            batchsize = inpshape[0]

            return [batchsize, z, fmapsout, y, x]


class shiftlayer(layer):
    """ Shift layer of the Shift-and-Stitch algorithm for lossless max-pooling """

    # TODO: Support for Sequential Data
    # Constructor
    def __init__(self, ds, bordermode='valid', makedecoder=True, inpshape=None):
        """
        :type ds: list or int
        :param ds: Downsampling (pooling) ratio

        :type bordermode: str
        :param bordermode: Border behaviour of the pooling sliding window. Can be set to 'valid' or 'same'.
                         ds must be odd if border mode is set to 'same'.

        :type makedecoder: bool
        :param makedecoder: Whether to activate decoder

        :type inpshape: tuple or list
        :param inpshape: Expected input shape.

        """

        # Initialize super class
        super(shiftlayer, self).__init__()

        # Check input
        ds = list(ds)
        ds = (ds * 2 if len(ds) == 1 else ds)

        # Assertions
        assert len(ds) <= 2, "3D shift-and-stitch not implemented yet."

        if bordermode is 'same' and not all([dsr % 2 == 1 for dsr in ds]):
            raise ValueError("ds must be odd for 'same' border handling to work.")

        # Meta
        self.ds = ds
        self.bordermode = bordermode
        self.decoderactive = makedecoder
        self.encoderactive = True
        self.dim = 2
        self.inpdim = 4

        # Dummy parameter lists
        self.params = []
        self.cparams = []

        if inpshape is None:
            self.inpshape = [None, ] * self.inpdim
        else:
            assert len(inpshape) == self.inpdim, "Length of inshape must equal the number of input dimensions."
            self.inpshape = inpshape

        self.outshape = self.inferoutshape()

        # Containers
        # Input
        self.x = T.tensor('floatX', [False, ] * self.inpdim, 'x:' + str(id(self)))
        # Output
        self.y = T.tensor('floatX', [False, ] * self.outdim, 'y:' + str(id(self)))

        # Generator parameters
        self.xshape = None

    # Feedforward through the layer
    def feedforward(self, inp=None):

        # Parse input
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        # Check if encoder active
        if not self.encoderactive:
            self.y = inp
            return inp

        # Get poolin'
        if self.bordermode is 'valid':
            inpooled = A.pool(input=inp, ds=self.ds, stride=(1, 1))
        elif self.bordermode is 'same':
            # Get padding size
            padding = [(dsr - 1) / 2 for dsr in self.ds]
            inpooled = A.pool(input=inp, ds=self.ds, stride=(1, 1), ignoreborder=True, padding=padding)
        else:
            raise NotImplementedError("Border mode: {} not implemented.".format(self.bordermode))

        # Set variables required for clipboard code to work  O:)
        # Number of images Per Batch
        npb = inpooled.shape[0]
        # Number of Feature mapS
        fs = inpooled.shape[1]
        # Edge Length: Y
        ely = inpooled.shape[2]
        # Edge Length: X
        elx = inpooled.shape[3]
        # DownSampling RATio
        dsrat = self.ds

        # Set xshape for makestitchlayer to work
        self.xshape = inpooled.shape

        # This is the line Lukas wants engraved in his headstone.
        # (Used with kind permission from Lukas Schott, lukas.schott@googlemail.com)
        self.y = inpooled. \
            reshape((1, npb * fs, ely, elx), ndim=4). \
            swapaxes(1, 3). \
            reshape((elx / dsrat[1], dsrat[1], ely / dsrat[0], dsrat[0], npb * fs), ndim=5). \
            swapaxes(0, 3). \
            reshape((dsrat[0] * dsrat[1], ely / dsrat[0], elx / dsrat[1], npb * fs), ndim=4). \
            swapaxes(1, 2). \
            swapaxes(1, 3). \
            reshape(((dsrat[0] * dsrat[1]) * npb, fs, ely / dsrat[0], elx / dsrat[1]), ndim=4)

        # Return
        return self.y

    # Method to auto-generate a corresponding stitch layer
    def makestitchlayer(self):
        assert None not in self.xshape, "Cannot make a valid stitch layer without shape information."
        return stitchlayer(self.ds, self.xshape)

# Infer output shapes
    def inferoutshape(self, inpshape=None, checkinput=False):
        if inpshape is None:
            inpshape = self.inpshape

        # Return input shape if encoder not active
        if not self.encoderactive:
            return inpshape

        if self.inpdim == 4:
            # Compute shape after pooling
            inpooledshape = inpshape if self.bordermode is "same" else [inpshape[n] - self.ds[[None, None, 0, 1][n]] + 1
                                                                        if n in [2, 3] else
                                                                        inpshape[n] for n in range(len(inpshape))]

            # Check input shape for any errors that might occur
            if checkinput:
                # Compute shape discrepancy
                shapediscr = \
                    [(inpooledshape[sid] % self.ds[kid]) if inpooledshape[sid] is not None else None for sid, kid in
                     zip([2, 3], [0, 1])]

                if any([discr is not 0 for discr in shapediscr]):
                    # Generate shape suggestions:
                    # After pooling:
                    suggpoolshape = inpooledshape[0:2] + \
                                    [shape - discr for shape, discr in zip(inpooledshape[2:], shapediscr)]
                    sugginpshape = suggpoolshape if self.bordermode is "same" else \
                        [suggpoolshape[n] + self.ds[[None, None, 0, 1][n]] - 1 if n in [2, 3] else suggpoolshape[n]
                         for n in range(len(suggpoolshape))]

                    # Generate error message and raise
                    errmsg = "Invalid shape for shift layer {}: " \
                             "{}. Suggested input shape: {} and shape after pooling: {}"\
                        .format(str(id(self)), inpshape, sugginpshape, suggpoolshape)
                    raise ValueError(errmsg)

                pass

            # Infer shape after shift
            y, x = [int(inpooledshape[sid] / self.ds[kid]) if inpooledshape[sid] is not None else None
                    for sid, kid in zip([2, 3], [0, 1])]
            fmapsout = inpooledshape[1]
            batchsize = np.prod(self.ds) * inpooledshape[0] if inpooledshape[0] is not None else None

            # Pre-Infer xshape (needed to initialize a partner stitch layer before this layer is fed forward)
            self.xshape = inpooledshape

            # Return
            return [batchsize, fmapsout, y, x]

        elif self.inpdim == 5:
            # TODO: Output shape for sequential data
            pass


class stitchlayer(layer):
    """ Stitch layer of the Shift-and-Stitch algorithm for lossless max-pooling """

    # Constructor
    def __init__(self, shiftds, shiftxshape, inpshape=None):
        """
        :type shiftds: list or int
        :param shiftds: Downsampling ratio used while shifting (with shift layer)

        :type shiftxshape: theano.tensor.var.TensorVariable or list
        :param shiftxshape: Shape of the downsampled tensor (i.e. of shift.x)

        :type inpshape: tuple or list
        :param inpshape: Expected shape of the input.

        """

        # Initialize super class
        super(stitchlayer, self).__init__()

        # Meta
        self.shiftds = shiftds
        self.shiftxshape = shiftxshape
        self.dim = 2
        self.inpdim = 4
        self.decoderactive = False
        self.encoderactive = True

        # Dummy parameters
        self.params = []
        self.cparams = []

        # Parse input shape
        if inpshape is None:
            self.inpshape = [None, ] * self.inpdim
        else:
            assert len(inpshape) == self.inpdim, "Length of inshape must equal the number of input dimensions."
            self.inpshape = inpshape

        self.outshape = self.inferoutshape()

        # Containers
        # Input
        self.x = T.tensor('floatX', [False, ] * self.inpdim, 'x:' + str(id(self)))
        # Output
        self.y = T.tensor('floatX', [False, ] * self.outdim, 'y:' + str(id(self)))

    # Feedforward through the layer
    def feedforward(self, inp=None):
        # Parse input
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        # Check if encoder active
        if not self.encoderactive:
            self.y = inp
            return inp

        # Set variables required for clipboard code to work  O:)
        # Number of images Per Batch
        npb = self.shiftxshape[0]
        # Number of Feature mapS. This might have changed since the shift layer was applied
        fs = inp.shape[1] if self.inpshape[1] is None else self.inpshape[1]
        # Edge Length: Y
        ely = self.shiftxshape[2]
        # Edge Length: X
        elx = self.shiftxshape[3]
        # DownSampling RATio
        dsrat = self.shiftds

        # Get stitchin'
        # So this line is basically undo-ing Lukas' line in shift.feedforward
        self.y = inp. \
            reshape((dsrat[0] * dsrat[1], npb * fs, ely / dsrat[0], elx / dsrat[1]), ndim=4). \
            swapaxes(1, 3). \
            swapaxes(1, 2). \
            reshape((dsrat[0], dsrat[1], ely / dsrat[0], elx / dsrat[1], npb * fs), ndim=5). \
            swapaxes(0, 3). \
            reshape((1, elx, ely, npb * fs), ndim=4). \
            swapaxes(1, 3). \
            reshape((npb, fs, ely, elx), ndim=4)

        # Return
        return self.y

    # Infer output shape
    def inferoutshape(self, inpshape=None, checkinput=False):
        if inpshape is None:
            inpshape = self.inpshape

        # Return input shape if encoder not active
        if not self.encoderactive:
            return inpshape

        if self.inpdim == 4:
            # Fetch
            y, x = [int(inpshape[sid] * self.shiftds[kid]) if inpshape[sid] is not None else None
                    for sid, kid in zip([2, 3], [0, 1])]
            fmapsout = inpshape[1]
            batchsize = inpshape[0] / np.prod(self.shiftds) if inpshape[0] is not None else None
            # Return
            return [batchsize, fmapsout, y, x]

        elif self.inpdim == 5:
            # TODO: Output shape for sequential data
            pass
        pass


class upsamplelayer(layer):
    """Unpool/upsample layer with or without interpolation"""

    def __init__(self, us, interpolate=False, allowsequences=True, fmapsin=None, activation=netools.linear(),
                 inpshape=None):
        """
        :type us: list or tuple
        :param us: Upsampling ratio.

        :type interpolate: bool
        :param interpolate: Whether to interpolate (i.e. convolve with a normalizede unit convolution kernel)

        :type allowsequences: bool
        :param allowsequences: Whether input can allowed to be a sequence (i.e. apply the upsampling framewise).
                               us must be [n, m, 1] where n and m are positive integers.

        :type fmapsin: int
        :param fmapsin: Number of input feature maps. Required for interpolation, but can also be infered from the
                        input shape.

        :type inpshape: list or tuple
        :param inpshape: Input shape
        :return:
        """

        # Construct superclass
        super(upsamplelayer, self).__init__()

        # Meta
        self.us = list(us)
        self.interpolate = interpolate
        self.allowsequences = allowsequences
        self.fmapsin = fmapsin
        self.activation = activation

        # Determine data and input dimensions
        self.inpdim = {2: 4, 3: 5}[len(us)]
        self.issequence = self.allowsequences and self.us[-1] == 1 if self.inpdim == 5 else False
        self.dim = 2 if (self.inpdim == 4 or self.issequence) else 3

        # Shape inference
        if inpshape is None:
            self.inpshape = [None, ] * self.inpdim
        else:
            self.inpshape = inpshape

        # Containers for input and output
        self.x = T.tensor('floatX', [False, ] * self.inpdim, name="x:" + str(id(self)))
        self.y = T.tensor('floatX', [False, ] * self.outdim, name="y:" + str(id(self)))

    def inferoutshape(self, inpshape=None, checkinput=True):
        if inpshape is None:
            inpshape = self.inpshape

        if checkinput:
            assert len(inpshape) == self.inpdim, "Length of the provided input shape does not match the " \
                                                 "number of input dimensions."

        if self.dim == 2:
            outshape = cp.copy(inpshape)
            outshape[-2:] = [shp * us if shp is not None else None for us, shp in zip(self.us[0:2], outshape[-2:])]
        elif self.dim == 3:
            outshape = cp.copy(inpshape)
            outshape[1] = outshape[1] * self.us[2] if outshape[1] is not None else None
            outshape[-2:] = [shp * us if shp is not None else None for us, shp in zip(self.us[0:2], outshape[-2:])]
        else:
            raise NotImplementedError

        return outshape

    def feedforward(self, inp=None):
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        if not self.encoderactive:
            self.y = inp
            return inp


        usd = A.unpool(inp, us=self.us, interpolate=self.interpolate, dim=self.dim, issequence=self.issequence)

        # This gets ugly: A.unpool does interpolation when us[0] == us[1]
        interpolate = self.interpolate and (self.us[0] != self.us[1] or self.dim == 3)

        # TODO: Move to backend.unpool
        # Interpolate if required
        if interpolate:
            # Make convolution kernel for interpolation.
            if self.dim == 2:
                # Fetch number of feature maps
                self.fmapsin = self.inpshape[(1 if self.inpdim == 4 else 2)] if self.fmapsin is None else self.fmapsin

                assert self.fmapsin is not None, "Number of input feature maps could not be inferred."

                # Make conv-kernels
                numconvker = netutils.setkernel(inds=[[fm, fm] for fm in range(self.fmapsin)],
                                                subkernel=(1./(self.us[0] * self.us[1])),
                                                out=np.zeros(
                                                    shape=(self.fmapsin, self.fmapsin, self.us[0], self.us[1]))).\
                    astype(th.config.floatX)
                convker = th.shared(value=numconvker)

            elif self.dim == 3:
                # Make convolution kernel for interpolation.
                # Fetch number of feature maps
                self.fmapsin = self.inpshape[(1 if self.inpdim == 4 else 2)] if self.fmapsin is None else self.fmapsin

                assert self.fmapsin is not None, "Number of input feature maps could not be inferred."

                # Make conv-kernels
                numconvker = netutils.setkernel(inds=[[fm, fm] for fm in range(self.fmapsin)],
                                                subkernel=(1./(self.us[0] * self.us[1] * self.us[2])),
                                                out=np.zeros(shape=(self.fmapsin, self.us[2], self.fmapsin,
                                                                    self.us[0], self.us[1]))).\
                    astype(th.config.floatX)
                convker = th.shared(value=numconvker)

            else:
                raise NotImplementedError

            # Convolve to interpolate
            usd = A.conv(usd, convker, convmode='same')

        self.y = usd
        return self.y


class softmax(layer):
    """ Framework embedded softmax function without learnable parameters """

    def __init__(self, dim, onehot=False, inpshape=None):
        """
        :type dim: int
        :param dim: Layer dimensionality. 1 for vectors, 2 for images and 3 for volumetric data

        :type onehot: bool
        :param onehot: Whether to encode one-hot for prediction

        :type inpshape: tuple or list
        :param inpshape: Shape of the expected input

        """

        # Initialize super class
        super(softmax, self).__init__()

        # Meta
        self.decoderactive = False
        self.encoderactive = True
        self.dim = dim
        self.onehot = onehot

        if self.dim == 1:
            self.inpdim = 2
        elif self.dim == 2:
            self.inpdim = 4
        elif self.dim == 3:
            self.inpdim = 5
        else:
            raise NotImplementedError('Invalid layer dimensionality: {}. Supported: 1D, 2D and 3D.'.format(self.dim))

        # Dummy parameters for compatibility with layertrain
        self.params = []
        self.cparams = []

        # Parse input shape
        if inpshape is None:
            self.inpshape = [None, ] * self.inpdim
        else:
            assert len(inpshape) == self.inpdim, "Length of inshape must equal the number of input dimensions."
            self.inpshape = inpshape

        self.outshape = self.inferoutshape()

        # Containers
        if self.dim == 1:
            # Input
            self.x = T.matrix('x:' + str(id(self)))
            # Output
            self.y = T.matrix('y:' + str(id(self)))
        elif self.dim == 2 or self.dim == 3:
            # Input
            self.x = T.tensor('floatX', [False, ] * self.inpdim, name='x:' + str(id(self)))
            # Output
            self.y = T.tensor('floatX', [False, ] * self.outdim, name='y:' + str(id(self)))
        else:
            raise NotImplementedError

    # Feedforward
    def feedforward(self, inp=None, predict=None):
        # inp is expected of the form
        #   inp.shape = (numimages, sigsin)

        # Parse Input
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        # Check if encoder active
        if not self.encoderactive:
            self.y = inp
            return inp

        if predict is None:
            predict = self.onehot

        self.y = A.softmax(inp, dim=self.dim, predict=predict, issequence=self.issequence)
        return self.y

    # Infer output shape
    def inferoutshape(self, inpshape=None, checkinput=False):
        if inpshape is None:
            inpshape = self.inpshape

        # Return input shape if encoder not active
        if not self.encoderactive:
            return inpshape

        # Set number of channels to 1 if onehot
        if self.onehot:
            inpshape[1] = 1

        return inpshape


class noiselayer(layer):
    """ General Noising Layer """

    # Constructor
    def __init__(self, noisetype=None, sigma=None, n=None, p=None, dim=2, thicken=True, makedecoder=False, rngseed=42,
                 inpshape=None):
        """
        :type noisetype: str
        :param noisetype: Possible keys: 'normal', 'binomial'.

        :type sigma: float
        :param sigma: std for normal noise

        :type n: float
        :param n: n for binomial (salt and pepper) noise

        :type p: float
        :param p: p for binomial (salt and pepper) noise (also the dropout amount)

        :type dim: int
        :param dim: Dimensionality of the layer

        :type thicken: bool
        :param thicken: (in Hinton speak) whether to divide the activations with the dropout amount (p)

        :type makedecoder: bool
        :param makedecoder: Noises in the decoder layer when set to True

        :type inpshape: tuple or list
        :param inpshape: Shape of the expected input
        """

        # Initialize super class
        super(noiselayer, self).__init__()

        # Meta
        self.thicken = thicken
        self.decoderactive = makedecoder
        self.encoderactive = True
        self.srng = RandomStreams(rngseed)

        # Dummy parameter list for compatibility with layertrain
        self.params = []
        self.cparams = []

        # Parse dimensionality
        self.dim = dim
        if self.dim == 1:
            self.inpdim = 2
        elif self.dim == 2:
            self.inpdim = 4
        elif self.dim == 3:
            self.inpdim = 5
        else:
            raise NotImplementedError('Invalid layer dimensionality: {}. Supported: 1D, 2D and 3D.'.format(self.dim))

        if not noisetype:
            self.noisetype = 'normal'
        else:
            self.noisetype = noisetype

        if not sigma:
            self.sigma = 0.2
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

        # Parse input shape
        if inpshape is None:
            self.inpshape = [None, ] * self.inpdim
        else:
            assert len(inpshape) == self.inpdim, "Length of inshape must equal the number of input dimensions."
            self.inpshape = inpshape

        self.outshape = self.inferoutshape()

        # Container for input (see feedforward() for input shapes) and output
        if self.dim == 2:
            # Input
            self.x = T.tensor('floatX', [False, False, False, False], name='x:' + str(id(self)))
            # Output
            self.y = T.tensor('floatX', [False, False, False, False], name='y:' + str(id(self)))

        elif self.dim == 3:
            # Input
            self.x = T.tensor('floatX', [False, False, False, False, False], name='x:' + str(id(self)))
            # Output
            self.y = T.tensor('floatX', [False, False, False, False, False], name='y:' + str(id(self)))

        elif self.dim == 1:
            # Input
            self.x = T.matrix('x:' + str(id(self)))
            # Output
            self.y = T.matrix('y:' + str(id(self)))

    # Feedforward
    def feedforward(self, inp=None):
        # Parse Input
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        # Check if encoder active
        if not self.encoderactive:
            self.y = inp
            return inp

        self.y = A.noise(inp, noisetype=self.noisetype, p=self.p, n=self.n, sigma=self.sigma, srng=self.srng)
        return self.y


# Batch Normalization Layer
class batchnormlayer(layer):
    """ Batch Normalization Layer con Nonlinear Activation. See Ioffe et al. (http://arxiv.org/abs/1502.03167). """

    def __init__(self, dim, momentum=0., axis=None, eps=1e-6, activation=netools.linear(gain=1.),
                 gamma=None, beta=None, makedecoder=True, inpshape=None):
        """
        :type dim: int
        :param dim: Dimensionality of the layer/network (2D or 3D)

        :type momentum: float
        :param momentum: Momentum of the moving average and std (over batches)

        :type axis: int
        :param axis: Axis over which to normalize

        :type eps: float
        :param eps: A small epsilon for numerical stability

        :type activation: callable
        :param activation: Activation function.

        :type gamma: callable or float or numpy.ndarray
        :param gamma: Default value for the scale factor gamma. Must be parsable as a ghost variable.

        :type beta: callable or float or numpy.ndarray
        :param beta: Default value for the shift amout beta. Must be parsable as a ghost variable.

        :type makedecoder: bool
        :param makedecoder: Whether to activate decoder

        :type inpshape: tuple or list
        :param inpshape: Expected input shape
        """

        # Init superclass
        super(batchnormlayer, self).__init__()

        # Meta
        self.momentum = momentum
        self.dim = dim
        self.eps = eps
        self.activation = activation
        self.encoderactive = True
        self.decoderactive = makedecoder
        # The input could be a sequence for all I care
        self.allowsequences = True

        # Debug
        self._epreshiftscale = None
        self._epreactivation = None
        self._dpostshiftscale = None
        self._dpreactivation = None

        # Parse input dimensions
        if self.dim == 2:
            self.inpdim = 4
        elif self.dim == 3:
            self.inpdim = 5
        else:
            raise NotImplementedError("Invalid layer dimension. Supported: 2 and 3.")

        # Parse input shape
        if inpshape is None:
            self.inpshape = [None, ] * self.inpdim
        else:
            assert len(inpshape) == self.inpdim, "Input shape must have exactly as many elements as the " \
                                                 "input dimension (4 for 2D, 5 for 3D)."
            self.inpshape = inpshape

        # Parse axis
        if axis is None:
            self.axis = {5: 2, 4: 1}[self.inpdim]
        else:
            assert axis in [1, 2]
            self.axis = axis

        # This layer is particularly hairy because the parameter shapes depend on the input shape. Inputs are generally
        # tensor variables while parameters are expected to be shared variables.
        # The user can be expected to provide the input shape of the first layer in a layertrain, but not of a layer
        # deep in the network.
        # Ghost variables to the rescue!
        # Since this is the first time ghost variables are being deployed, here are some general ground rules:
        #   1. The layer must clean up it's own mess: that includes appropriately updating ghost variable shapes and
        #      instantiating new instances of ghost variable when necessary. This prevents the ghost variable mess
        #      from spilling over to other parts of the project.
        #   2. All that can be expected from layertrain (or a more abstract class) is that the instantiate() method of
        #      the ghost variables be called while feeding forward.

        # Get normalization shape

        # Parse default values for ghost params
        if beta is None:
            beta = lambda shp: np.zeros(shape=shp)
        else:
            if callable(beta):
                # Beta is good
                pass
            elif isinstance(beta, float):
                # Convert floating point number to a callable returning number * ones matrix
                beta = (lambda pf: lambda shp: pf * np.ones(shp, dtype=th.config.floatX))(beta)
            else:
                raise NotImplementedError("Beta must be a callable or a floating point number.")

        if gamma is None:
            gamma = lambda shp: np.ones(shape=shp)
        else:
            if callable(gamma):
                # Gamma good
                pass
            elif isinstance(gamma, float):
                gamma = (lambda pf: lambda shp: pf * np.ones(shp, dtype=th.config.floatX))(gamma)
            else:
                raise NotImplementedError("Beta must be a callable or a floating point number.")

        # Function to compute ghost parameter shape given the input shape
        self.getghostparamshape = lambda shp: [shp[self.axis], ]

        # Ghost params
        self.beta = netutils.ghostvar(shape=self.getghostparamshape(self.inpshape),
                                      value=beta,
                                      name='bnbeta:' + str(id(self)),
                                      shared=True)
        self.gamma = netutils.ghostvar(shape=self.getghostparamshape(self.inpshape),
                                       value=gamma,
                                       name='bngamma:' + str(id(self)),
                                       shared=True)

        # Ghost connectivity params
        self.betac = netutils.ghostvar(shape=self.getghostparamshape(self.inpshape),
                                       value=1.,
                                       name='bnbetac:' + str(id(self)),
                                       shared=True)
        self.gammac = netutils.ghostvar(shape=self.getghostparamshape(self.inpshape),
                                        value=1.,
                                        name='bngammac:' + str(id(self)),
                                        shared=True)

        # Gather ghost parameters and cparameters
        self.ghostparams = [self.gamma, self.beta]
        self.ghostcparams = [self.gammac, self.betac]
        # Gather parameters and cparameters. Right now, self.params = self.ghostparams, but that should change in the
        # feedforward() method.
        self._params = [self.gamma, self.beta]
        self._cparams = [self.gammac, self.betac]

        # Initialize state variables
        self.runningmean = 0.
        self.runningstd = 1.

        # Set state
        self.state = [self.runningmean, self.runningstd]

        # Container for input (see feedforward() for input shapes) and output
        # Input
        self.x = T.tensor('floatX', [False, ] * self.inpdim, name='x:' + str(id(self)))
        # Output
        self.y = T.tensor('floatX', [False, ] * self.outdim, name='y:' + str(id(self)))

    def feedforward(self, inp=None, activation=None):
        # Parse input
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        # Identity if encoder is not active
        if not self.encoderactive:
            return inp

        # Parse activation
        activation = self.activation if activation is None else activation

        # Instantiate params from ghost state
        self.instantiate()

        __old__ = False

        if not __old__:
            if not self.testmode:
                # Training graph
                # Compute batch norm
                y, bm, bstd = A.batchnorm(self.x, self.gamma, self.beta, gammamask=self.gammac, betamask=self.betac,
                                          axis=self.axis, eps=self.eps, dim=self.dim, issequence=self.issequence)
                # Add running mean and running std updates to updaterequests
                updreq = dict(self.updaterequests)
                updreq.update({self.runningmean: self.momentum * self.runningmean + (1 - self.momentum) * bm,
                               self.runningstd: self.momentum * self.runningstd + (1 - self.momentum) * bstd})
                self.updaterequests = updreq.items()

            else:
                y, bm, bstd = A.batchnorm(self.x, self.gamma, self.beta, mean=self.runningmean, std=self.runningstd,
                                          gammamask=self.gammac, betamask=self.betac, axis=self.axis, eps=self.eps,
                                          im=self.dim, issequence=self.issequence)

            self.y = y
            return self.y

        if __old__:
            # Determine reduction axes
            redaxes = range(self.inpdim)
            redaxes.pop(self.axis)
            broadcastaxes = redaxes

            # Compute mean and standard deviation
            batchmean = T.mean(inp, axis=redaxes)
            batchstd = T.sqrt(T.var(inp, axis=redaxes) + self.eps)

            # Broadcast running mean and std, batch mean and std
            broadcastpattern = ['x'] * self.inpdim
            broadcastpattern[self.axis] = 0

            rm = self.runningmean.dimshuffle(*broadcastpattern)
            rstd = self.runningstd.dimshuffle(*broadcastpattern)
            bm = batchmean.dimshuffle(*broadcastpattern)
            bstd = batchstd.dimshuffle(*broadcastpattern)

            if not self.testmode:
                # Place update requests. Remember that feedforward could have been called before (i.e. to use dict updates).
                updreq = dict(self.updaterequests)
                updreq.update({self.runningmean: self.momentum * self.runningmean + (1 - self.momentum) * batchmean,
                               self.runningstd: self.momentum * self.runningstd + (1 - self.momentum) * batchstd})
                self.updaterequests = updreq.items()

                # Normalize input.
                norminp = (inp - bm) / (bstd)
                # For debug
                self._epreshiftscale = norminp
            else:
                norminp = (inp - rm) / (rstd)
                # For debug
                self._epreshiftscale = norminp

            # TODO: Gradient clips
            # Shift and scale
            # Broadcast params
            gammabc = self.gamma.dimshuffle(*broadcastpattern)
            betabc = self.beta.dimshuffle(*broadcastpattern)
            gammacbc = self.gammac.dimshuffle(*broadcastpattern)
            betacbc = self.betac.dimshuffle(*broadcastpattern)

            self._epreactivation = tho.maskgradient(gammabc, gammacbc) * norminp + tho.maskgradient(betabc, betacbc)

            # Activate
            self.y = activation(self._epreactivation)

            # Return
            return self.y

    def instantiate(self):
        # Instantiate with the machinery in layer
        super(batchnormlayer, self).instantiate()

        # Gamma and beta are no longer ghost variables, but real theano variables.
        # Rebind gamma and beta to the instantiated variables
        self.gamma, self.beta = self.params
        self.gammac, self.betac = self.cparams

        # Initialize running mean and running std variables. This should be possible because inpshape is now known.
        self.runningmean = th.shared(np.zeros(shape=self.getghostparamshape(self.inpshape), dtype=th.config.floatX),
                                     name='rmean:' + str(id(self)))
        self.runningstd = th.shared(np.ones(shape=self.getghostparamshape(self.inpshape), dtype=th.config.floatX),
                                    name='rstd:' + str(id(self)))
        # TODO: Have runningmean and runningstd as ghost variables
        self.state = [self.runningmean, self.runningstd]

    def inferoutshape(self, inpshape=None, checkinput=False):
        # Parse
        if inpshape is None:
            inpshape = self.inpshape

        # Return input shape if encoder not active
        if not self.encoderactive:
            return inpshape

        # Batch normalization does not change the input shape anyway.
        return inpshape


# LSTM Layer
class convlstm(layer):
    def __init__(self, forgetgate, inpgate, outgate, inpmodgate=None, inneractivation=T.tanh,
                 peepholes=True, returnsequence=True, persistentmemory=True, truncatebptt=-1, inpshape=None):

        """
        :type inpgate: convlayer or layertrain
        :param inpgate: Input gate. Could be shallow (a single convlayer) or deep (a layertrain)

        :type inpmodgate: convlayer or layertrain
        :param inpmodgate: Input modulation gate. Could be shallow (convlayer) or deep (layertrain).

        :type forgetgate: convlayer or layertrain
        :param forgetgate: Forget gate. Could be shallow (convlayer) or deep (layertrain)

        :type outgate: convlayer or layertrain
        :param outgate: Output gate. Could be shallow (convlayer) or deep (layertrain)

        :type peepholes: bool
        :param peepholes: Whether to include peephole connections

        :type returnsequence: bool
        :param returnsequence: Whether to return a sequence (of the same length as the input sequence)

        :type persistentmemory: bool
        :param persistentmemory: Whether to use persistent memory. This requires cellshape be given in full.

        :type truncatebptt: int
        :param truncatebptt: Where to truncate BPTT. Setting to -1 results in full gradient calculation
        """

        # Init super class
        super(convlstm, self).__init__()

        # Meta
        self.decoderactive = False
        self.encoderactive = True
        self.dim = 2  # Support only for 2D networks, for now
        self.inpdim = 5  # Support only for sequential data
        self.peepholes = peepholes
        self.persistentmemory = persistentmemory
        self.inneractivation = inneractivation
        self.truncatebptt = truncatebptt
        self.allowsequences = True
        self.issequence = True
        self.returnsequence = returnsequence
        self.recurrencenumber = 2

        # Assert all necessary gates are either convlayers or layertrain
        assert all([isinstance(gate, convlayer) or isinstance(gate, layertrain)
                    for gate in [inpgate, forgetgate, outgate]]), "Forget-, Input- and Output-Gates must be " \
                                                                  "convlayers or layertrains."

        self.forgetgate = forgetgate
        self.inpgate = inpgate
        self.outgate = outgate
        # Inpmodgate might still be None, in which case we consider it coupled to the forget gate
        self.inpmodgate = inpmodgate
        # Store in a convienient gates attribute
        self.gates = [self.forgetgate, self.inpgate, self.outgate] + ([] if self.inpmodgate is None else
                                                                      [self.inpmodgate])

        # TODO: Deactivate gradient masking on all convlayers

        # Parse input shape
        if inpshape is None:
            self.inpshape = [None, ] * self.inpdim
        else:
            assert len(inpshape) == self.inpdim, "Length of inshape must equal the number of input dimensions."
            self.inpshape = inpshape

        self.outshape = self.inferoutshape()
        self.cellshape, _ = self.inferstateshape()
        self.numcells = self.cellshape[1]

        assert self.numcells is not None, "Number of memory cells not provided."

        # Build LSTM state
        self.h0 = None
        self.C0 = None
        self._state = []
        self.buildstate()

        # Get update requests from all gates
        self.updaterequests = [request for gate in self.gates for request in gate.updaterequests]

        # Build a list of parameters (aggregated from all gates)
        self._params = [param for gate in self.gates for param in gate.params]
        self._cparams = [cparam for gate in self.gates for cparam in gate.cparams]

        # Containers for input and output
        # Input
        self.x = T.tensor('floatX', [False, ] * self.inpdim, name='x:' + str(id(self)))
        # Output
        self.y = T.tensor('floatX', [False, ] * self.outdim, name='y:' + str(id(self)))

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        # Make sure value is a list
        if not isinstance(value, list):
            value = [value]
        # Make sure all elements in value are numerical (numpy ndarrays)
        try:
            value = [(param.get_value() if not isinstance(param, np.ndarray) else param) for param in value]
        except AttributeError:
            value = [(param.eval() if not isinstance(param, np.ndarray) else param) for param in value]
        # Apply parameters with applyparams
        self.applyparams(params=value)

    @property
    def cparams(self):
        return self._cparams

    @cparams.setter
    def cparams(self, value):
        # Make sure value is a list
        if not isinstance(value, list):
            value = [value]
        # Make sure all elements in value are numerical (numpy ndarrays)
        try:
            value = [(param.get_value() if not isinstance(param, np.ndarray) else param) for param in value]
        except AttributeError:
            value = [(param.eval() if not isinstance(param, np.ndarray) else param) for param in value]
        # Apply parameters with applyparams
        self.applyparams(cparams=value)

    @property
    def state(self):
        return self._state

    @layer.inpshape.setter
    def inpshape(self, value):
        # Check if shapelock is active
        if self.shapelock:
            warn("Can not set input shape. Disarm shape lock and try again.")
            return

        # Assign new inpshape and run shape inference functions
        self._inpshape = value
        self.outshape = self.inferoutshape(inpshape=value)
        self.cellshape, _ = self.inferstateshape(inpshape=value)
        self.outdim = len(self.outshape)

        # State variables need rebuilding as well.
        self.buildstate()

    # Step forward in time.
    def step(self, inp, htm1, Ctm1):
        # inp is assumed of the shape (numbatches, numchannels, y, x). Do not use directly with 5D tensors.
        # Concatenate input and previous hidden layer (htm1) along the channel axis
        x = T.concatenate([inp, htm1] + ([Ctm1] if self.peepholes else []), axis=1)

        # Run through forget, hidden and output gates
        forget = self.forgetgate.feedforward(inp=x)
        input = self.inpgate.feedforward(inp=x)
        inputmod = (1. - forget if self.inpmodgate is None else self.inpmodgate.feedforward(inp=x))
        output = self.outgate.feedforward(inp=x)

        # Forget, learn and tell
        Ct = Ctm1 * forget + Ctm1 * inputmod * input
        ht = output * self.inneractivation(Ct)
        # Return output (at time t) first, and then the cell state. The order is very important.
        return ht, Ct

    # Feedforward method
    def feedforward(self, inp=None):
        # Parse input
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        # inp.shape = (numbatches, T, numchannels, y, x). Push T axis to front because scan loops over the 0-th axis
        inp = inp.dimshuffle(1, 0, 2, 3, 4)

        # Initialize hidden layer and cell state. For persistent memory, init as the saved activations from the
        # previous function run
        if self.persistentmemory:
            self.instantiate()
            h0, C0 = self.h0, self.C0
        else:
            h0, C0 = [st.instantiate(shape=inp.shape[1:]) for st in self.state]

        # Scan applies the step function (self.step) recurrently.
        [ht, ct], _ = th.scan(self.step, sequences=inp, outputs_info=[{"initial": h0},
                                                                      {"initial": C0}],
                              truncate_gradient=self.truncatebptt)

        # Check if a sequence is requested
        if not self.returnsequence:
            hT, cT = ht[-1], ct[-1]
        else:
            hT, cT = ht.dimshuffle(1, 0, 2, 3, 4), ct.dimshuffle(1, 0, 2, 3, 4)

        hT.name, cT.name = "hT:" + str(id(self)), "cT:" + str(id(self))

        # Request update if memory persistent
        if self.persistentmemory:
            # Ideally, we could do something like:
            # self.updaterequests += [(self.h0, hT), (self.C0, cT)]
            # but that would re-append if the feedforward method is run again. What we want to keep is the update
            # request of the current run and get rid of the update request tuples from the previous run. dict's update
            # method does just that, so:
            updict = dict(self.updaterequests)
            updict.update({self.h0: ht[-1], self.C0: ct[-1]})
            self.updaterequests = updict.items()

        # Set output and return
        self.y = hT
        return self.y

    def applyparams(self, params=None, cparams=None):

        # Params is a list with parameters of all gates. Parse how many parameters belong to which gates
        lenlist = [len(gate.params) for gate in self.gates]

        # Declare cursors on params
        cursorstart = cursorstop = 0

        # Loop over gates
        for gatenum, gate in enumerate(self.gates):

            # Skip if there are no parameters to be applied
            if lenlist[gatenum] == 0:
                continue

            # Figure out where to place the stop cursor
            cursorstop = cursorstart + lenlist[gatenum]

            # Fetch parameters and cparameters to apply
            if params is not None:
                gateparams = params[cursorstart:cursorstop]
            else:
                gateparams = None

            if cparams is not None:
                gatecparams = cparams[cursorstart:cursorstop]
            else:
                gatecparams = None

            # Apply
            gate.applyparams(params=gateparams, cparams=gatecparams)

            # Place start cursor for the next iteration
            cursorstart = cursorstop

    def inferoutshape(self, inpshape=None, checkinput=True):
        # Parse
        if inpshape is None:
            inpshape = self.inpshape

        if not self.encoderactive:
            return inpshape

        inpshape = list(inpshape)

        # Inpshape is 5D sequential, i.e. inpshape.shape = (numbatches, T, numchannels, row, col).
        # Obtain the shape of gate input
        gateinpshape = inpshape[0:1] + inpshape[2:]

        # Cell shape must equal output shape of the forget gate.
        # We can, however, use the given input shape to infer the forget gate's output shape although it eats the input
        # concatenated with the previous hidden layer. This is possible by design, because the gate's inferoutshape
        # method does not necessarily use the number of input channels to compute the output shape.
        cellshape = self.forgetgate.inferoutshape(gateinpshape, checkinput=False)

        # Compute the shape of the layer output (as inferoutshape of layertrain would expect it)
        outshape = cellshape[0:1] + inpshape[1:2] + cellshape[1:]

        if checkinput and gateinpshape[1] is not None:
            # Check if cellshape plays good with inpshape.
            gateinpshape[1] += (2 if self.peepholes else 1) * cellshape[1]

            # This would raise an error if input shape doesn't checkout and checkinput is set to true.
            fgateoutshape = self.forgetgate.inferoutshape(gateinpshape, checkinput=True)

            # Check if input gate checks out
            igateoutshape = self.inpgate.inferoutshape(gateinpshape, checkinput=True)
            assert igateoutshape == cellshape, "Input gate output shape must equal cell shape."

            # Check if output gate checks out
            ogateoutshape = self.outgate.inferoutshape(gateinpshape, checkinput=True)
            assert ogateoutshape == cellshape, "Output gate output shape must equal cell shape."

            # Check if input modulation gate checks out
            if self.inpmodgate is not None:
                imgateoutshape = self.inpmodgate.inferoutshape(gateinpshape, checkinput=True)
                assert imgateoutshape == cellshape, "Output modulation gate output shape must equal cell shape."

        return outshape

    def inferstateshape(self, inpshape=None):
        # Parse
        if inpshape is None:
            inpshape = self.inpshape

        inpshape = list(inpshape)

        # Inpshape is 5D sequential, i.e. inpshape.shape = (numbatches, T, numchannels, row, col).
        # Obtain the shape of gate input
        gateinpshape = inpshape[0:1] + inpshape[2:]

        # Cell shape must equal output shape of the forget gate.
        # We can, however, use the given input shape to infer the forget gate's output shape although it eats the input
        # concatenated with the previous hidden layer. This is possible by design, because the gate's inferoutshape
        # method does not necessarily use the number of input channels to compute the output shape.
        cellshape = self.forgetgate.inferoutshape(gateinpshape, checkinput=False)

        # Return shapes of the two state variables (which happen to be equal for LSTMs)
        return cellshape, cellshape

    def buildstate(self):
        # Use ghost variables to build a hidden state. This is not straight-forward because the state shape depends
        # on the input shape, which may not be known a priori (at least when the layer is constructed), but at the
        # same time, we may require the state variables to persist over function calls (when persistentmemory = True).
        # Recall that theano shared variables persist over function calls but requires the shape to be known.
        self.h0 = netutils.ghostvar(shape=self.cellshape,
                                    value=lambda initshape: np.random.uniform(size=initshape,
                                                                              low=-1.,
                                                                              high=1.).astype(th.config.floatX),
                                    shared=self.persistentmemory, name='h0:' + str(id(self)))
        self.C0 = netutils.ghostvar(shape=self.cellshape,
                                    value=lambda initshape: np.random.uniform(size=initshape,
                                                                              low=-1.,
                                                                              high=1.).astype(th.config.floatX),
                                    shared=self.persistentmemory, name='C0:' + str(id(self)))

        self._state = [self.h0, self.C0]

    def instantiate(self):
        msg = "Can not instantiate variable {} of ConvLSTM layer due to missing shape (cellshape) information."
        assert None not in self.h0.shape, msg.format("h0")
        assert None not in self.C0.shape, msg.format("C0")

        # Instantiate and reassign to _state
        self.h0 = self.h0.instantiate() if isinstance(self.h0, netutils.ghostvar) else self.h0
        self.C0 = self.C0.instantiate() if isinstance(self.C0, netutils.ghostvar) else self.C0
        self._state = [self.h0, self.C0]

    def __pow__(self, other, modulo=None):
        if isinstance(other, recurrentchain):
            return recurrentchain(chain=[self] + other.chain,
                                  persistentmemory=self.persistentmemory or other.persistentmemory,
                                  truncatebptt=max(self.truncatebptt, other.truncatebptt)
                                  if self.truncatebptt is not -1 or other.truncatebptt is not -1 else -1)
        elif isinstance(other, layer):
            return recurrentchain(chain=[self, other], persistentmemory=self.persistentmemory,
                                  truncatebptt=self.truncatebptt)
        else:
            raise NotImplementedError("Can only multiply with a recurrentchain or a layer object.")


# Gaussian denoising layer for ladder networks
class denoiselayer(layer):
    """Gaussian Denoising Layer (as used in Ladder Networks, see: http://arxiv.org/abs/1507.02672"""
    def __init__(self, ai=None, inpshape=None):
        """
        :type ai: tuple
        :param ai: a_i's for the "miniature networks" (see http://arxiv.org/abs/1507.02672)

        :type inpshape: list
        :param inpshape: Input shape (can't init ghost-variables otherwise)
        """
        super(denoiselayer, self).__init__()

        # 2D implementation only
        self.dim = [2, 2]
        self.inpdim = [4, 4]
        self.numinp = 2
        self.numout = 1

        if inpshape is None:
            self.inpshape = [[None, ] * indim for indim in self.inpdim]
        else:
            self.inpshape = inpshape

        numparams = 10
        aI = [None, ] * numparams if ai is None else ai

        self._params = []
        self._cparams = []

        # Function to get ghost variable shapes given the input shape
        self.getghostparamshape = lambda shp: shp[0]

        # Init ghost variables
        for i, ai in zip(range(1, 11), aI):
            if i in [2, 7]:
                # Init with ones
                initai = lambda shp: np.ones(shape=shp)
            else:
                # Init with zeros
                initai = lambda shp: np.zeros(shape=shp)

            # Init connectivity variables
            initcai = lambda shp: np.ones(shape=shp)

            ai = netutils.ghostvar(shape=self.inpshape[0],
                                   value=initai,
                                   name="a{}:".format(i) + str(id(self)),
                                   shared=True)

            aic = netutils.ghostvar(shape=self.inpshape[0],
                                    value=initcai,
                                    name="a{}c:".format(i) + str(id(self)),
                                    shared=True)

            # Set params
            self.__setattr__('a{}'.format(i), ai)
            self._params.append(self.__getattribute__('a{}'.format(i)))
            # Set cparams
            self.__setattr__('a{}c'.format(i), aic)
            self._cparams.append(self.__getattribute__('a{}c'.format(i)))

        # Containers for input and output
        self.x = [T.tensor('floatX', [False, ] * indim, name='x{}:'.format(inpnum) + str(id(self)))
                  for inpnum, indim in enumerate(self.inpdim)]
        self.y = T.tensor('floatX', [False, ] * self.outdim, name='y:' + str(id(self)))

    def feedforward(self, inp=None):
        # Parse
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        # Provide shapes to ghost variables
        for p, cp in zip(self.params, self.cparams):
            if isinstance(p, netutils.ghostvar):
                p.shape = self.inpshape[0]
            if isinstance(cp, netutils.ghostvar):
                cp.shape = self.inpshape[0]

        # Changes in inpshape are mirrored to ghostvarables by inpshape.setter of the superclass
        self.instantiate()

        for i, realparam, realcparam in zip(range(1, 11), self.params, self.cparams):
            self.__setattr__('a{}'.format(i), realparam)
            self.__setattr__('a{}c'.format(i), realcparam)

        assert not any([isinstance(p, netutils.ghostvar) for p in self.params + self.cparams]), "Ghost variables " \
                                                                                                "were not instantiated."

        # Fetch inputs
        # u: Decoder output from the previous layer
        # zc: Corrupted activation from the corresponding encoder
        u, zc = inp

        mu = tho.maskgradient(self.a1, self.a1c) * T.nnet.sigmoid(tho.maskgradient(self.a2, self.a2c) * u +
                                                                  tho.maskgradient(self.a3, self.a3c)) + \
             tho.maskgradient(self.a4, self.a4c) * u + tho.maskgradient(self.a5, self.a5c)
        v = tho.maskgradient(self.a6, self.a6c) * T.nnet.sigmoid(tho.maskgradient(self.a7, self.a7c) * u +
                                                                 tho.maskgradient(self.a8, self.a8c)) + \
            tho.maskgradient(self.a9, self.a9c) * u + tho.maskgradient(self.a10, self.a10c)

        # Compute denoised estimate
        zest = (zc - mu) * v + mu

        # Return
        self.y = zest
        return self.y

    def inferoutshape(self, inpshape=None, checkinput=True):
        if inpshape is None:
            inpshape = self.inpshape

        if checkinput:
            assert len(inpshape) == 2, "Too many or too few inputs."
            assert [ishp0 == ishp1 if None not in [ishp0, ishp1] else True
                    for ishp0, ishp1 in zip(inpshape[0], inpshape[1])]

        outshape = inpshape[0]

        return outshape