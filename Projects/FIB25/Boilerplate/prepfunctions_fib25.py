__doc__ = "Preprocessing functions for CREMI."

import numpy as np
from scipy.ndimage import convolve
from scipy.signal import convolve as convolve2
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates, zoom

import Antipasti.prepkit as pk
import Antipasti.pykit as pyk

import itertools as it


def prepfunctions():
    # Define preprocessing functions
    # Function convert segmentation to membrane segmentation\
    def seg2membrane():

        def fun(batch):

            def labels2membrane(im):
                # Compute gradients in x, y and threshold gradient magnitude
                gx = convolve(np.float32(im), np.array([-1., 0., 1.]).reshape(1, 3))
                gy = convolve(np.float32(im), np.array([-1., 0., 1.]).reshape(3, 1))
                return np.float32((gx**2 + gy**2) > 0)

            # Loop over images in batch, compute gradient and threshold
            return np.array([np.array([labels2membrane(im)
                                       for im in chim]) for chim in batch]).astype('float32')

        return fun

    # Function to apply exp-euclidean distance transform
    def disttransform(gain):

        def fun(batch):
            # Invert batch
            batch = 1. - batch
            # Merge batch and channel dimensions
            bshape = batch.shape
            batch = batch.reshape((bshape[0] * bshape[1], bshape[2], bshape[3]))
            # Distance transform by channel
            transbatch = np.array([np.exp(-gain * distance_transform_edt(img)) for img in batch])
            # Reshape batch and return
            return transbatch.reshape(bshape)

        return fun

    # 3D to channel distributed 2D
    def time2channel(batch):
        return np.array([sample.squeeze() for sample in batch])

    # Trim out all channels except the center
    def trim2center(batch):
        if batch.shape[1] % 2 == 0:
            return batch[:, (batch.shape[1] / 2):(batch.shape[1] / 2 + 1), ...]
        else:
            return batch[:, (batch.shape[1] / 2):(batch.shape[1] / 2 + 1), ...]

    # Function to add the complement of a batch as an extra channel
    def catcomplement(batch):
        # Compute complement
        cbatch = 1. - batch
        return np.concatenate((batch, cbatch), axis=1)

    def elastictransform(sigma, alpha, rng=np.random.RandomState(42), affgraph=False):
        # Define function on image
        def _elastictransform3D(*images):
            # Take measurements
            imshape = images[0].shape
            # Make random fields
            dx = rng.uniform(-1, 1, imshape) * alpha
            dy = rng.uniform(-1, 1, imshape) * alpha
            # Smooth dx and dy
            sdx = gaussian_filter(dx, sigma=sigma, mode='reflect')
            sdy = gaussian_filter(dy, sigma=sigma, mode='reflect')
            # Make meshgrid
            x, y = np.meshgrid(np.arange(imshape[1]), np.arange(imshape[0]))
            # Distort meshgrid indices (invert if required)
            distinds = (y + sdy).reshape(-1, 1), (x + sdx).reshape(-1, 1)

            # Map cooordinates from image to distorted index set
            transformedimages = [map_coordinates(image, distinds, mode='reflect', order=1).reshape(imshape)
                                 for image in images]

            return transformedimages

        def _func2(batches):

            # Get the number of channels for all elements in batches
            lenlist = [batch.shape[1] for batch in batches]

            # Pre-init
            outs = []

            for samples in zip(*batches):
                # Convert samples to a list of lists
                samplelist = [list(sample) for sample in samples]
                # Concatenate lists to a big list
                imglist = reduce(lambda x, y: x + y, samplelist)
                # Run elastic transformation
                transformedimglist = _elastictransform3D(*imglist)
                # Structure list of transformed images to samples
                transformedsamplelist = [np.array(sample) for sample in pyk.unflatten(transformedimglist, lenlist)]
                outs.append(transformedsamplelist)

            # len(outs) = 2 if batchsize == 2
            outs = [np.array(out) for out in zip(*outs)]

            return outs

        return _func2

    def randomocclude(size=32, frequency=0.2, rng=np.random.RandomState(42), noise=True, gloc=0.,
                      gstd=1., interweave=True):

        def occlude(*images):
            imshape = images[0].shape
            # Get random mask
            randmaskshape = [ishp / size for ishp in imshape]
            dsrandmask = rng.binomial(1, frequency, size=randmaskshape)
            randmask = np.repeat(np.repeat(dsrandmask, size, axis=0), size, axis=1)
            if noise:
                # Make white noise
                wnoise = rng.normal(loc=gloc, scale=gstd, size=imshape)
            else:
                # Make black patch
                wnoise = np.zeros(shape=imshape)
            # Occlude
            processedimages = []
            for image in images:
                image[randmask == 1.] = wnoise[randmask == 1.]
                processedimages.append(image)
            # Return processed images
            return pyk.delist(processedimages)

        def _func(batch):
            batchX, batchY, batchYs = batch
            # Occlude images in batchX
            batchXc = pk.image2batchfunc(occlude)(batchX.copy())
            if interweave:
                # Interweave
                nbatchX = np.zeros(shape=(2 * batchX.shape[0],) + batchX.shape[1:])
                nbatchX[0::2, ...] = batchX
                nbatchX[1::2, ...] = batchXc
                nbatchY = np.repeat(batchY, 2, axis=0)
                nbatchYs = np.repeat(batchYs, 2, axis=0)
            else:
                nbatchX = batchXc
                nbatchY = batchY
                nbatchYs = batchYs
            # Return
            return nbatchX, nbatchY, nbatchYs

        return _func

    def multiscale():
        def _func(batch):
            batchX, batchY, batchYs = batch
            # Make downsampling functions
            ds2 = pk.image2batchfunc(lambda im: zoom(im, zoom=0.5, order=1, mode='reflect'))
            ds4 = pk.image2batchfunc(lambda im: zoom(im, zoom=0.25, order=1, mode='reflect'))
            ds8 = pk.image2batchfunc(lambda im: zoom(im, zoom=0.125, order=1, mode='reflect'))
            # Downsample and return
            return batchX, batchY, ds8(batchY), ds4(batchY), ds2(batchY), batchY
        return _func

    # Function to drop the central frame of the raw data
    def dropslice(proba=0.5):
        def _func(batch):
            # Fetch X batch from the batch tuple
            batchX = batch[0]
            # Get rid of the central frame
            if proba < np.random.uniform():
                batchX[:, 1, ...] = 0.
            return (batchX, ) + batch[1:]
        return _func

    # Function to drop any (but just one) random frame of raw data
    def rdropslice(proba=0.5):
        def _func(batch):
            # Fetch X batch
            batchX = batch[0]
            # Get number of channels
            nc = batchX.shape[1]
            if proba < np.random.uniform():
                for batchind in range(batchX.shape[0]):
                    # Get a random integer corresponding to the object to drop
                    drop = np.random.randint(0, nc)
                    # Drop object
                    batchX[batchind, drop, ...] = 0.
            return (batchX,) + batch[1:]
        return _func

    # Function to drop multiple slices at random
    def mdropslice(proba=0.5, numdrop=0.5):
        def _func(batch):
            batchX = batch[0]
            nc = batchX.shape[1]
            # Number of slices to drop
            nd = np.floor(nc * numdrop)
            # Drop slices
            if proba < np.random.uniform():
                for batchind in range(batchX.shape[0]):
                    # Get an array of random integers
                    drops = np.random.randint(0, nc, size=(nd,))
                    batchX[batchind, drops, ...] = 0.
            return (batchX,) + batch[1:]
        return _func

    # Flip slices vertically
    def randomflipz(pairflip=False, debug=False):

        def _flipz(batch):

            if debug:
                print("Input shape is: {}".format(batch.shape))

            if not pairflip:
                if debug:
                    print("Pathway 1.")
                batch = batch[:, ::-1, ...]
            else:
                if debug:
                    print("Pathway 2.")
                assert batch.shape[1] == 6
                batch = np.concatenate((batch[:, :3, ...][:, ::-1, ...], batch[:, 3:, ...][:, ::-1, ...]), axis=1)

            if debug:
                print("Output shape is: {}".format(batch.shape))

            return batch

        def _func(batches):
            if np.random.uniform() > 0.5:
                return tuple([_flipz(bat) for bat in batches])
            else:
                return batches

        return _func

    # Random rotate
    def randomrotate():
        def _rotatetensor(tensor, k):
            return np.array([np.array([np.rot90(chim, k) for chim in sample]) for sample in tensor])

        def _func(batches):
            # Pick a random k to rotate
            k = np.random.randint(0, 4)
            return [_rotatetensor(batch, k) for batch in batches]

        return _func

    # Random flip
    def randomflip():
        # Define batch callables
        batchfunctions = {'lr': pk.image2batchfunc(np.fliplr),
                          'ud': pk.image2batchfunc(np.flipud),
                          'lrtd': pk.image2batchfunc(lambda im: np.fliplr(np.flipud(im)))}

        def _func(batches):
            # assert mode in ['lr', 'td', 'lrtd']
            funcid = ['lr', 'ud', 'lrtd'][np.random.randint(3)]
            func = batchfunctions[funcid]
            return [func(batch) for batch in batches]

        return _func

    # Make weight maps
    def wmapmaker(eps=1e-6):
        # Make filter function and batchify
        @pk.image2batchfunc
        def box(img):
            # Smooth along x
            smx = convolve(img, np.array([1., 1., 1.]).reshape(1, 3))
            sm = convolve(smx, np.array([1., 1., 1.]).reshape(3, 1))
            return sm.astype('float32')

        def _func(batches):
            # Fetch X and Y batches
            batchX, batchY = batches[0:2]
            # Find patches of zeros. This can be done by convolving the raw data with a box filter and thresholding at
            # a very small value.
            batchW = (box(batchX) > eps).astype('float32')
            return batchW

        return _func

    return vars()
