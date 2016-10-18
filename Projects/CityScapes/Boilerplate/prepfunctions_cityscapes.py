"""Preprocessing functions for CityScapes"""

import random

import numpy as np
from skimage.transform import resize


def prepfunctions():

    def wmapmaker():
        """Extract weight map from the label tensor."""
        def _func(batches):
            rawbatch, labelbatch = batches
            # Make wmap from labelbatch
            newlabelbatch = labelbatch[:, 0:-1, ...]
            wmap = labelbatch[:, -1:, ...]
            return rawbatch, newlabelbatch, wmap

        return _func

    def patchmaker(patchsize):
        """Extract patches from image and batch."""
        def _func(batches):

            rawbatch = batches[0]
            # Take measurements
            bs, nc, nrow, ncol = rawbatch.shape

            # Define a function to return a random patch
            def samplepatch():
                # Get patch size
                ypatchsize, xpatchsize = patchsize
                # Compute bounds
                yminstart = 0
                xminstart = 0
                ymaxstart = nrow - ypatchsize
                xmaxstart = ncol - xpatchsize
                # Sample
                ystart = random.randint(yminstart, ymaxstart)
                xstart = random.randint(xminstart, xmaxstart)
                # Make slice
                return slice(ystart, ystart + ypatchsize), slice(xstart, xstart + xpatchsize)

            # Get a different patch for every batch
            sls = [samplepatch() for _ in range(bs)]
            # Crop and return
            return tuple([np.array([chim[(slice(0, None),) + sl] for chim, sl in zip(batch, sls)])
                          for batch in batches])

        return _func

    def ds(ratio):
        """Downsample by a given ratio."""
        def _func(batches):
            out = tuple([batch[:, :, ::ratio[0], ::ratio[1]] for batch in batches])
            return out

        return _func

    def patchds(outimshape, maxpatchsize, minpatchsize):
        """Extracts a patch while simultaneously downsampling at random."""

        # Define a function to return a random patch
        def samplepatch(patchsize, nrow, ncol):
            # Get patch size
            ypatchsize, xpatchsize = patchsize if isinstance(patchsize, tuple) else (patchsize, patchsize)
            # Compute bounds
            yminstart = 0
            xminstart = 0
            ymaxstart = nrow - ypatchsize
            xmaxstart = ncol - xpatchsize
            # Sample
            ystart = random.randint(yminstart, ymaxstart)
            xstart = random.randint(xminstart, xmaxstart)
            # Make slice
            return slice(ystart, ystart + ypatchsize), slice(xstart, xstart + xpatchsize)

        def _func(batches):

            # Take measurements
            # Number of rows and cols in the source image
            bs, _, nrowsrc, ncolsrc = batches[0].shape

            # Sample a patchsize between max and min patchsize for all batches
            patchsizes = [random.randint(minpatchsize, maxpatchsize) for _ in range(bs)]

            # Sample patches
            sls = [samplepatch(patchsize, nrowsrc, ncolsrc) for patchsize in patchsizes]

            # Make resize function on channel
            _resize = lambda chim: np.array([resize(im, output_shape=outimshape, preserve_range=True, order=0)
                                             for im in chim])

            # Process and return
            return tuple([np.array([_resize(chim[(slice(0, None),) + sl]) for chim, sl in zip(batch, sls)])
                          for batch in batches])

        # Done
        return _func

    def normalize(meanvals=None):
        # Normalize like the VGG expects it.
        def _func(batchX):

            # Compute meanvals if none provided
            _meanvals = meanvals
            if np.array(_meanvals) == np.array(None):
                _meanvals = batchX.mean(axis=(0, 2, 3))
            # Reshape
            _meanvals = _meanvals.reshape(1, 3, 1, 1)

            # Subract mean
            batchX = batchX - _meanvals

            # Put the batch back together and return
            return batchX

        return _func

    return vars()

