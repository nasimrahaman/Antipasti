# TODO: A wrapper around a generator that checks if the weights are all zero;

import numpy as np

def skipper(batches):
    # The weight maps should have been computed by now
    batchW = batches[2]
    # Check if batch is all 0
    return not np.allclose(batchW, 0)

