"""
A module with a package-wide random number generator,
used for weight initialization and seeding noise layers.
This can be replaced by a :class:`numpy.random.RandomState` instance with a
particular seed to facilitate reproducibility.

Borrowed initially from https://github.com/Lasagne/Lasagne/blob/master/lasagne/random.py
and modified for our use cases
"""

import numpy as np
import torch


_numpy_rng = np.random


def get_numpy_rng(seed=None):
    if seed is None:
        return _numpy_rng
    else:
        return np.random.RandomState(seed)


def set_seed(seed, seed_all_gpus=False):
    global _numpy_rng
    _numpy_rng = np.random.RandomState(seed=seed)
    if seed_all_gpus:
        torch.cuda.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


def randomized_indices(n_indices, arange_start=0, arange_stop=None,
                       replace=False, rng=None):
    if arange_stop is None:
        arange_stop = n_indices

    if rng is None:
        rng = _numpy_rng

    return rng.choice(np.arange(arange_start, arange_stop), size=n_indices,
                      replace=replace)
