# coding=utf-8
"""
Utility functions for the Pytorch library
"""
import logging

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import FloatTensor
from torch import LongTensor
from torch.autograd import Variable

cuda_tensor_types = (torch.cuda.ByteTensor,
                     torch.cuda.CharTensor,
                     torch.cuda.DoubleTensor,
                     torch.cuda.FloatTensor,
                     torch.cuda.HalfTensor,
                     torch.cuda.IntTensor,
                     torch.cuda.LongTensor,
                     torch.cuda.ShortTensor)

cpu_tensor_types = (torch.ByteTensor,
                    torch.CharTensor,
                    torch.DoubleTensor,
                    torch.FloatTensor,
                    torch.HalfTensor,
                    torch.IntTensor,
                    torch.LongTensor,
                    torch.ShortTensor)

logger = logging.getLogger(__name__)


def is_cuda_type(value):
    value_ = value
    if isinstance(value, Variable):
        value_ = value_.data
    if isinstance(value_, cpu_tensor_types):
        return False
    elif isinstance(value_, cuda_tensor_types):
        return True
    else:
        raise ValueError("Unknown type of input: {}".format(type(value_)))


def long_variable_from_numpy(numpy_matrix, cuda=False):
    """
    Convert integer numpy matrix to a Pytorch tensor for indexing operations

    :param numpy_matrix: an integer-type numpy matrix 
    :type numpy_matrix: numpy.ndarray

    :param cuda:  if True, output is GPU-type tensor, else CPU-type tensor
    :type cuda: bool

    :returns: A LongTensor which is used in Pytorch for indexing other Tensors
    :rtype: torch.LongTensor
    """
    out = Variable(LongTensor(numpy_matrix.astype(np.int64)))
    if cuda:
        out = out.cuda()
    return out


def float_variable_from_numpy(numpy_matrix, cuda=False):
    """
    :param numpy_matrix: an float-type numpy matrix 
    :type numpy_matrix: numpy.ndarray

    :param cuda:  if True, output is GPU-type tensor, else CPU-type tensor
    :type cuda: bool

    :returns: A FloatTensor
    """
    out = Variable(FloatTensor(numpy_matrix.astype(np.float32)))
    if cuda:
        out = out.cuda()
    return out


def strip_variable(func):
    def out_func(*args, **kwargs):
        out_args = []
        for value in args:
            if isinstance(value, Variable):
                out_args.append(value.data)
            else:
                out_args.append(value)

        out_kwargs = {}
        for name, value in kwargs.items():
            if isinstance(value, Variable):
                out_kwargs[name] = value.data
            else:
                out_kwargs[name] = value
        return func(*out_args, **out_kwargs)

    return out_func


@strip_variable
def compute_accuracy(y_pred, y_true, scale=100.):
    y_pred = y_pred.cpu().squeeze().numpy().argmax(axis=1)
    y_true = y_true.cpu().squeeze().numpy()
    return np.equal(y_pred, y_true).sum() / y_pred.shape[0] * scale


@strip_variable
def compute_f1(y_pred, y_true, mode="macro", scale=100.):
    y_pred = y_pred.cpu().squeeze().numpy().argmax(axis=1)
    y_true = y_true.cpu().squeeze().numpy()
    return f1_score(y_true, y_pred, average=mode) * scale


