# coding=utf-8
"""
Utility functions for the Pytorch library
"""
import logging

import numpy as np
from pyromancy import constants
import torch
from torch import FloatTensor
from torch import LongTensor
from torch.autograd import Variable

logger = logging.getLogger(__name__)


def get_sequence_lengths(numpy_matrix):
    """
    :param numpy_matrix: A matrix where each row is a 0-padded sequence
    :type numpy_matrix: numpy.ndarray

    :returns: A vector of the lengths of the sequences
    """
    return np.array([len(x.nonzero()[0]) for x in numpy_matrix])


def long_variable_from_numpy(numpy_matrix, cuda=False, volatile=False):
    """
    Convert integer numpy matrix to a Pytorch tensor for indexing operations

    :param volatile:
    :param numpy_matrix: an integer-type numpy matrix
    :type numpy_matrix: numpy.ndarray

    :param cuda:  if True, output is GPU-type tensor, else CPU-type tensor
    :type cuda: bool

    :returns: A LongTensor which is used in Pytorch for indexing other Tensors
    :rtype: torch.LongTensor
    """
    # noinspection PyArgumentList
    out = Variable(LongTensor(numpy_matrix.astype(np.int64)),
                   volatile=volatile)
    if cuda:
        out = out.cuda()
    return out


def float_variable_from_numpy(numpy_matrix, cuda=False, volatile=False):
    """
    :param volatile:
    :param numpy_matrix: an float-type numpy matrix
    :type numpy_matrix: numpy.ndarray

    :param cuda:  if True, output is GPU-type tensor, else CPU-type tensor
    :type cuda: bool

    :returns: A FloatTensor
    """
    # noinspection PyArgumentList
    out = Variable(FloatTensor(numpy_matrix.astype(constants.NUMPY_FLOAT_X)),
                   volatile=volatile)
    if cuda:
        out = out.cuda()
    return out


def numpy_from_torch(torch_var_or_tensor):
    """
    This function will move the data to cpu, squeeze it, and strip the
    torch Variable if it is wrapping the data.

    :param torch_var_or_tensor: a torch numeric instance.

    :returns: the numpy data inside the torch numeric instance
    """
    torch_var_or_tensor = torch_var_or_tensor.cpu().squeeze()
    if isinstance(torch_var_or_tensor, Variable):
        torch_var_or_tensor = torch_var_or_tensor.data
    return torch_var_or_tensor.numpy()


def compute_accuracy(y_pred, y_true, scale=100.):
    y_pred = numpy_from_torch(y_pred).argmax(axis=1)
    y_true = numpy_from_torch(y_true)
    return np.equal(y_pred, y_true).sum() / float(y_pred.shape[0]) * scale


def compute_f1(y_pred, y_true, mode="macro", scale=100.):
    from sklearn.metrics import f1_score
    y_pred = numpy_from_torch(y_pred).argmax(axis=1)
    y_true = numpy_from_torch(y_true)
    return f1_score(y_true, y_pred, average=mode) * scale
