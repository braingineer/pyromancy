# coding=utf-8

import torch
from torch.nn import functional as F

from pyromancy import pyromq
from pyromancy.utils.torchutils import compute_accuracy, compute_f1


class MetricGroup(pyromq.PublisherGroup):
    """
    The MetricPublisher serves as the base class for all Metric and Loss managers

    The MetricPublisher is intended to operate in the following way:
        1. init: Instantiate it.
        2. add_metric: Add metric objects to it, specifying the target data you'd like metrics
            computed on.
        3. set_router: Add to a training driver, who will set itself as the router.
        4. compute: The training driver then calls compute for each batch computation
            and each metric computes its value for the provided data and predictions.
            The augmented function is created in the add_metric method.
    """

    def __init__(self, name="metric_group", channel_name="metric_events",
                 broker=None):
        """
        This __init__ is implemented so that it can set the default name and
        channel_name.
        """
        super(MetricGroup, self).__init__(name=name,
                                          channel_name=channel_name,
                                          broker=broker)

    def compute(self, in_dict, out_dict, data_type):
        """
        Args:
            in_dict (dict): The inputs to the network in dictionary form
                `{ input_name: input_value, ... }`

            out_dict (dict): Predictions from the network
                `{ output_name: output_value, ... }`

            data_type (str): String specifying what type of data this is
                `data_type in ('train', 'eval', 'dev', 'test', ...)`

        Returns:
            None
        """
        metrics = list(self.members_by_target.get(data_type, []))
        metrics += list(self.members_by_target.get("*", []))

        for metric in metrics:
            _ = metric(in_dict, out_dict, data_type)


# Individual Metrics/Losses

class SingleIOMetric(pyromq.ComputePublisher):
    def __init__(self, name, target_name, output_name,
                 channel_name='metric_events', broker=None):
        super(SingleIOMetric, self).__init__(name, channel_name=channel_name,
                                             broker=broker)
        self.target_name = target_name
        self.output_name = output_name


class Accuracy(SingleIOMetric):
    def compute(self, in_dict, out_dict):
        """
        TODO
        """
        y_true = in_dict[self.target_name]
        y_pred = out_dict[self.output_name]
        return compute_accuracy(F.softmax(y_pred), y_true)


class SiameseAccuracy(pyromq.ComputePublisher):
    def __init__(self, name, output1_name, output2_name, labels_name, threshold=0.,
                 channel_name='metric_events', broker=None):
        super(SiameseAccuracy, self).__init__(name, channel_name=channel_name,
                                              broker=broker)
        self.output1_name = output1_name
        self.output2_name = output2_name
        self.labels_name = labels_name
        self.threshold = threshold

        if str(threshold) not in self.name:
            self.name += '@' + str(threshold)

    def compute(self, in_dict, out_dict):
        """
        TODO
        """
        y1 = out_dict[self.output1_name]
        y2 = out_dict[self.output2_name]
        labels = in_dict[self.labels_name]
        sim = F.cosine_similarity(y1, y2)

        classified_true = torch.gt(sim, self.threshold)
        correctly_true = torch.eq(classified_true, labels.byte())
        return torch.mean(correctly_true.float())


class MacroF1(SingleIOMetric):
    def compute(self, in_dict, out_dict):
        """
        TODO

        :return:
        """
        y_true = in_dict[self.target_name]
        y_pred = out_dict[self.output_name]
        return compute_f1(F.softmax(y_pred),
                          y_true,
                          mode="macro")


class MicroF1(SingleIOMetric):
    def compute(self, in_dict, out_dict):
        """
        TODO

        :return:
        """
        y_true = in_dict[self.target_name]
        y_pred = out_dict[self.output_name]
        return compute_f1(F.softmax(y_pred),
                          y_true,
                          mode="micro")
