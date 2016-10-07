"""Preprocessing functions for CityScapes"""

import random
import numpy as np


def prepfunctions():

    def wmapmaker():

        def _func(batches):
            rawbatch, labelbatch = batches
            # Make wmap from labelbatch
            newlabelbatch = labelbatch[:, 1:, ...]
            wmap = labelbatch[:, 0:1, ...]
            return rawbatch, newlabelbatch, wmap

        return _func

    def patchmaker(patchsize):

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

        def _func(batches):
            out = tuple([batch[:, :, ::ratio[0], ::ratio[1]] for batch in batches])
            return out

        return _func

    return vars()

