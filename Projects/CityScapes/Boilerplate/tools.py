import random

import numpy as np
import h5py as h5

import Antipasti.netdatakit as ndk
import Antipasti.trainkit as tk
import Antipasti.prepkit as pk


def path2dict(path):
    if isinstance(path, str):
        return tk.yaml2dict(path)
    elif isinstance(path, dict):
        return path
    else:
        raise NotImplementedError


def ffd(dictionary, key, default=None):
    if key in dictionary.keys():
        return dictionary[key]
    else:
        return default


class csfeeder(ndk.datafeeder):
    """Dedicated Feeder for CityScapes."""
    def __init__(self, feederconfig, preptrain=None):
        """
        :type feederconfig: str or dict
        :param feederconfig: (Path to) Feeder configuration.

        :type preptrain: Antipasti.prepkit.preptrain
        :param preptrain: Train of preprocessing functions.
        """
        # Meta
        self.feederconfig = path2dict(feederconfig)

        # Data info
        self.numsamples = None

        # Open files and datasets
        self.openfile = None
        self.h5dsets = None
        self.open()

        # Preptrain
        self.preptrain = pk.preptrain([]) if preptrain is None else preptrain

        # Iterator
        self.iterator = None
        self.restartgenerator()

    def open(self):
        """Open H5 file and read datasets."""
        self.openfile = h5.File(self.feederconfig['h5path'], 'r+')
        self.h5dsets = {key: self.openfile[self.feederconfig['pathh5'][key]]
                        for key in self.feederconfig['pathh5'].keys()}
        # Check if the sample sizes checkout
        assert self.h5dsets['raw'].shape[0] == self.h5dsets['labels'].shape[0]
        self.numsamples = self.h5dsets['raw'].shape[0]

    def batchstream(self):

        # Get batch slices
        batchslices = [slice(i, i + self.feederconfig['batchsize'])
                       for i in range(0, self.numsamples, self.feederconfig['batchsize'])]

        # Shuffle batches if required
        if self.feederconfig['shuffle']:
            random.shuffle(batchslices)

        # Go!
        for sl in batchslices:
            # Fetch raw and label batches
            rawbatch = np.asarray(self.h5dsets['raw'][sl])
            labelbatch = np.asarray(self.h5dsets['labels'][sl])
            # Run through preptrain
            pbatch = self.preptrain((rawbatch, labelbatch))
            # Yield
            yield pbatch

    def next(self):
        return self.iterator.next()

    def restartgenerator(self):
        self.iterator = self.batchstream()

    def cleanup(self):
        """Clean up any opened files, etc."""
        self.openfile.close()
