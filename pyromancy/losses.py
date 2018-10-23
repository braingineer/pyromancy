# coding=utf-8
# noinspection PyPackageRequirements

from pyromancy import pyromq
import torch
from torch.nn import functional as F


class LossGroup(pyromq.PublisherGroup):
    def __init__(self, optimizer, grad_clip_norm=-1, **kwargs):
        super(LossGroup, self).__init__(**kwargs)

        self.optimizer = optimizer
        self._accumulated_loss = None
        self.grad_clip_norm = grad_clip_norm

    def zero_grad(self):
        """
        TODO
        """
        self.optimizer.zero_grad()
        self._accumulated_loss = None

    def _accumulate_loss(self, computed_loss):
        """
        TODO
        """
        if self._accumulated_loss is None:
            self._accumulated_loss = computed_loss
        else:
            self._accumulated_loss += computed_loss

    def step(self, post_zero_grad=True):
        """
        TODO
        """
        self._accumulated_loss.backward()
        if self.grad_clip_norm > 0:
            for group in self.optimizer.param_groups:
                torch.nn.utils.clip_grad_norm(group['params'],
                                              self.grad_clip_norm)
        self.optimizer.step()
        if post_zero_grad:
            self.zero_grad()

    def compute(self, in_dict, out_dict, data_type):
        for loss in self.members_by_target[data_type]:
            loss_result = loss(in_dict, out_dict, data_type)
            self._accumulate_loss(loss_result['pytorch_value'])


class Loss(pyromq.ComputePublisher):
    def __init__(self, name, weight=1.0, **kwargs):
        super(Loss, self).__init__(name, **kwargs)
        self.weight = weight


class NegativeLogLikelihood(Loss):
    def __init__(self, name, target_name, output_name, **kwargs):
        super(NegativeLogLikelihood, self).__init__(name, **kwargs)
        self.target_name = target_name
        self.output_name = output_name

    def compute(self, in_dict, out_dict):
        """
        TODO
        """
        y_true = in_dict[self.target_name]
        y_pred = out_dict[self.output_name]
        computed_loss = self.weight * F.cross_entropy(y_pred, y_true)
        return computed_loss


class SiameseLoss(Loss):
    def __init__(self, name, output1_name, output2_name, labels_name, **kwargs):
        super(SiameseLoss, self).__init__(name, **kwargs)
        self.output1_name = output1_name
        self.output2_name = output2_name
        self.labels_name = labels_name

    def compute(self, in_dict, out_dict):
        """
        TODO
        """
        y1 = out_dict[self.output1_name]
        y2 = out_dict[self.output2_name]
        labels = in_dict[self.labels_name]
        return self.weight * F.cosine_embedding_loss(y1, y2, labels)
