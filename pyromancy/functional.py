"""
A set of functions for doing functional-style data streaming for
deep learning with pytorch
"""
import itertools

import numpy as np
import torch
from torch.autograd import Variable


def make_batcher(batch_size, strict_batching=True):
    """
    Convert a single-item generator into a generator of batched items

    Args:
        batch_size (int): number of items to batch together per iteration
        strict_batching (bool): If True, will not yield batches that are smaller
            than batch_size.  This can happen at the end of a generator when
            there are not enough items left.

    Returns:
        A function which takes as input a generator and outputs a new generator
    """

    def inner(single_item_generator):
        while True:
            batch = list(itertools.islice(single_item_generator, 0, batch_size))
            if len(batch) == 0 or (strict_batching and len(batch) < batch_size):
                break
            yield batch

    return inner


def make_batch_collater(convert_to_torch=True, use_cuda=False,
                        make_volatile=False):
    """
    In the case where there is a list of tuples, collate them.

    This is primarily used for dataset iteration where the dataset will yield
        out datapoints, one at a time.  The `batcher` function above is used
        to group those data points, and this collater will then group each of
        the tuples and apply necessary numerical conversions to it (such as
        converting to pytorch's primitives)

    Args:
        convert_to_torch (bool): default True; If True, will convert numpy
            objects into PyTorch Variables

        use_cuda (bool): default False; if True, will move the PyTorch tensors
            to the GPU. Necessarily requires `convert_to_torch` to be True

        make_volatile (bool): default False; If True, will call the volatile
            flag on the PyTorch Variable, which renders it unable to hold a
            gradient computation.  This is useful when making predictions with
            the model and the gradient is not needed (saves RAM)
    """

    def inner(batch_generator):
        for batch in batch_generator:
            if isinstance(batch[0], dict):
                # case: each batch is a dictionary
                batch_out = {}
                keys = batch[0].keys()
            elif isinstance(batch[0], (tuple, list)):
                # case: each batch is a tuple/list of values
                batch_out = [None] * len(batch[0])
                keys = range(len(batch[0]))
            else:
                raise RuntimeError("incorrect values in batch generator; "
                                   "`batch_collater` expects a generator of "
                                   "dictionaries or tuples/lists so that it can"
                                   "collate the values")

            # collate among the keys (either dict keys or list indices)
            for key in keys:
                # stack for numeric computing
                value = np.stack([b[key] for b in batch])
                # convert to torch.. usually the case :)
                if convert_to_torch:
                    value = Variable(torch.from_numpy(value),
                                     volatile=make_volatile)
                    if use_cuda:
                        value = value.cuda()
                batch_out[key] = value
            yield batch_out

    return inner


def compose_functions(*sequence_of_functions):
    """
    Construct a composed function to apply the sequence of functions in order

    Examples:
        # Example 1: modifying values

        def double(x):
            return x * 2

        def add5(x):
            return x + 5

        composed_func1 = function_composer(double, add5)
        composed_func2 = function_composer(add5, double)

        print(composed_func1(10))
        print(composed_func2(10))

        # Example 2: modifying generators or sequences

        def double_each(generator):
            for x in generator:
                yield x * 2

        def add5_to_each(generator):
            for x in generator:
                yield x + 5

        composed_func1 = function_composer(double_each, add5_to_each)
        composed_func2 = function_composer(add5_to_each, double_each)

        print(list(composed_func1(range(10))))
        print(list(composed_func2(range(10))))

    Args:
        *sequence_of_functions: comma separated arguments are unpacked in this
            function
    """

    def inner(arg):
        for func in sequence_of_functions:
            arg = func(arg)
        return arg

    return inner
