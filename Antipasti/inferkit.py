import os
import numpy as np
import itertools as it
import multiprocessing as mp

#  Inference Engine
def infer(datain, model, preptrain, posptrain, windowsin, windowsout, dataout=None, verbose=True, device=None):
    """
    Infer on a block of data (`datain`) and write results to another block (inplace, `dataout`) with an Antipasti model
    (`model`).

    :type datain: numpy.ndarray
    :param datain: Input block.

    :type model: Antipasti.netarchs.model
    :param model: Model to infer with.

    :type preptrain: Antipasti.prepkit.preptrain
    :param preptrain: Train of preprocessing functions to convert the input data slice to appropriate dimensions for
                      model to process.

    :type posptrain: Antipasti.prepkit.preptrain
    :param posptrain: Train of postprocessing functions to process the output of the model, i.e. convert it to the
                      right shape.

    :type windowsin: generator or dict
    :param windowsin: Sliding window generator or configuration dict for input data. See kwargs in
                      Antipasti.netdatautils.slidingwindowslices for more on how to populate the configuration dict.

    :type windowsout: generator or dict or callable
    :param windowsout: Sliding window on the output data. If callable: must accept input sliding window as an input to
                       produce an output sliding window. If dict: kwargs for Antipasti.netdatautils.slidingwindowslices.
                       If generator: Must yield the next slice.

    :type dataout: numpy.ndarray
    :param dataout: Output block, to be written inplace.

    :return: Processed block.
    """

    def smartprint(msg):
        if verbose:
            print(msg)

    # Set up device
    if device is not None:
        smartprint("Process {}: Trying to initialize device {}...".format(os.getpid(), device))
        from theano.sandbox.cuda import use
        use(device=device)

    # Imports. All imports from now happen on 'device'.
    import Antipasti.netdatautils as ndu
    import Antipasti.netkit as nk
    import Antipasti.netarchs as na

    # Build model if required
    if not isinstance(model, na.model) and callable(model):
        smartprint("Process {}: Building Model...".format(os.getpid()))
        model = model()

    # Compile model if required
    if model.classifier is None:
        smartprint("Process {}: Compiling Model...".format(os.getpid()))
        model.compile(what='inference')

    # Build output data container if required
    if dataout is None:
        dataout = np.zeros_like(datain).astype('float32')

    normalization = np.zeros_like(dataout).astype('float32')

    # Fetch sliding window generator if windowsin is a dict
    if isinstance(windowsin, dict):
        windowsin = ndu.slidingwindowslices(**windowsin)

    # Fetch sliding window generator if windowsout is a dict
    if isinstance(windowsout, dict):
        windowsout = ndu.slidingwindowslices(**windowsout)

    # Loop over windowsin,
    for window in windowsin:
        smartprint("Process {}: Processing window {}...".format(os.getpid(), window))
        # fetch input volume,
        inp = datain[window]
        # apply preptrain,
        inp = preptrain(inp)
        # apply model,
        out = model.classifier(inp)
        # apply posptrain,
        out = posptrain(out)

        # fetch output slice from windowsout,
        if callable(windowsout):
            outwindow = windowsout(window)
        elif hasattr(windowsout, 'next'):
            outwindow = windowsout.next()
        else:
            raise RuntimeError("windowsout must be a callable or a generator or a dict.")

        # add to dataout and increment normalization tensor
        dataout[outwindow] += out
        normalization[outwindow] += 1.

    # Get rid of zeros from normalization
    normalization[normalization == 0.] = 1.
    # Normalize dataout
    dataout = dataout/normalization
    # Return dataout
    return dataout


# Dispatch Engine
class dispatcher(object):
    def __init__(self, data, dispatchconfig, numdevices=1):

        # Meta
        self.data = data
        self.dispatchconfig = dispatchconfig
        self.numdevices = numdevices
        self.outdata = None

        # Convert config to a usable list of slices
        list2slicelist = lambda indlist: [slice(0, indlist[0])] + \
                                         [slice(indlist[n], indlist[n+1]) for n in xrange(len(indlist[1:-1]))] + \
                                         [slice(indlist[-1], None)]
        dispatchslices = [[slice(0, None)] if conf == 'x' else list2slicelist(conf) for conf in self.dispatchconfig]

        # Build the actual chunks to dispatch
        self.dispatchchunks = list(it.product(*dispatchslices))


    def run(self, f, *fargs, **fkwargs):
        # Preallocate output data
        dout = np.zeros_like(self.data)

        if self.numdevices > 1:
            # Multiprocessing interface
            # Build pool of workers (Pool.imap isn't suited for this)
            processes = [mp.Process(target=lambda *args, **kwargs: f(*args, device=device, **kwargs), args=fargs,
                                    kwargs=fkwargs)
                         for device in range(self.numdevices)]
            # TODO
            raise NotImplementedError
        else:
            # Single process interface
            for chunkslice in self.dispatchchunks:
                out = f(self.data[chunkslice], *fargs, **fkwargs)
                dout[chunkslice] = out

        # Set and Return
        self.outdata = dout
        return dout

    def flush(self):
        self.outdata = None




# Test
if __name__ == '__main__':
    pass