# coding=utf-8
"""
"""
import collections
from datetime import datetime
import json
import logging
import os
import sqlite3
import time

import numpy as np
import pandas as pd
from pandas.io.sql import DatabaseError
from pyromancy import pyromq
from pyromancy.utils import torchutils
from pyromancy import constants
import torch
from torch.optim import lr_scheduler

logger = logging.getLogger(__name__)

TS_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


class Echo(pyromq.Subscriber):
    """
    The most basic Subscriber.  It is mainly used for debugging.  For actual
    printed statements during the training, please use LogSubscriber.
    """

    def training_events(self, message):
        print(message)

    def metric_events(self, message):
        print(message)

    def actuator_events(self, message):
        print(message)


class Stopwatch(pyromq.PublishingSubscriber):
    """
    Stopwatch will listen to info events and measure the timing.

    It is also a basic example of a PublishingSubscriber.
    """

    def __init__(self, name, frequency=1, channel_name='info_events',
                 broker=None):
        """
        Args:
            name (str)
            frequency (int)
            channel_name (str)
            broker (pyromq.Broker or None)
        """

        super(Stopwatch, self).__init__(name=name, channel_name=channel_name,
                                        broker=broker)
        self.total_batches = 0
        self.count = 0
        self.start_time = 0
        self.frequency = frequency

    def info_events(self, message):
        if message['name'] == "data.total_batches":
            self.total_batches = message["value"]
        elif message['name'] == "data.start":
            self.count = 0
            self.start_time = time.time()

    @staticmethod
    def _format_stopwatch(timestamp):
        return "{:>4d} min, {:>02d} sec".format(int(timestamp) // 60,
                                                int(timestamp) % 60)

    def training_events(self, message):
        if message['name'] == pyromq.MESSAGES.BATCH_END:
            self.count += 1
            time_elapsed = time.time() - self.start_time
            time_per_batch = time_elapsed / self.count
            time_remaining = (self.total_batches - self.count) * time_per_batch

            if self.broker is not None and self.count % self.frequency == 0:
                elapsed_string = self._format_stopwatch(time_elapsed)
                remaining_string = self._format_stopwatch(time_remaining)
                publish_format = "[{:>15} elapsed][{:>15} remaining]"
                publish_string = publish_format.format(elapsed_string,
                                                       remaining_string)
                self.publish(self.name, publish_string)


class Aggregate(pyromq.Subscriber):
    """
    Aggregate listens to metric events and aggregates them.
    """

    def __init__(self):
        self.num_values = {}
        self.aggregate_values = {}

    def metric_events(self, message):
        if message['name'] not in self.num_values:
            self.num_values[message['name']] = 0
            self.aggregate_values[message['name']] = 0.0

        self.num_values[message['name']] += 1
        self.aggregate_values[message['name']] += message['value']

    def reset(self):
        self.num_values = {k: 0 for k in self.num_values.keys()}
        self.aggregate_values = {k: 0 for k in self.aggregate_values.keys()}


class ChronometerSubscriber(pyromq.Subscriber):
    """
    The ChronometerSubscriber subscribes to training events and info events and
    records timing information
    """

    def __init__(self, experiment_name, trial_name, db_path, architecture,
                 dataset, ablation, batch_size):
        """
        :param experiment_name:
        :param trial_name:
        :param db_path:
        :param architecture:
        :param dataset:
        :param ablation:
        :param batch_size:
        """
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.conn = sqlite3.connect(db_path)

        self.architecture = architecture
        self.dataset = dataset
        self.ablation = ablation
        self.batch_size = batch_size

        self.epoch_index = 0
        self.batch_index = 0

        self.training_start = None
        self.epoch_start = None
        self.batch_start = None

        self._db_entries = []

    def add_entry(self, name, duration):
        """Adds an entry into the ongoing entries; entries pushed at end of epoch

        :param name:
        :param duration:
        """
        self._db_entries.append({"experiment_name": self.experiment_name,
                                 "trial_name": self.trial_name,
                                 "event_name": name,
                                 "duration": duration,
                                 "architecture": self.architecture,
                                 "dataset": self.dataset,
                                 "ablation": self.ablation,
                                 "batch_size": self.batch_size,
                                 "epoch_index": self.epoch_index,
                                 "batch_index": self.batch_index,
                                 "timestamp": datetime.now().strftime(
                                     TS_FORMAT)})

    def training_events(self, message):
        """Catch a training event and measure its running time

        :param message:
        """
        message_name = message['name']

        if message_name == pyromq.MESSAGES.TRAINING_START:
            self.training_start = time.time()
        elif message_name == pyromq.MESSAGES.EPOCH_START:
            self.epoch_start = time.time()
        elif message_name == pyromq.MESSAGES.BATCH_START:
            self.batch_start = time.time()
        elif message_name == pyromq.MESSAGES.TRAINING_END:
            if self.training_start is not None:
                self.add_entry('training', time.time() - self.training_start)
                self.push_to_db()
                # reset training start.. though technically this is end of everything
                self.training_start = None
        elif message_name == pyromq.MESSAGES.EPOCH_END:
            if self.epoch_start is not None:
                self.add_entry('epoch', time.time() - self.epoch_start)
                self.push_to_db()
                # reset epoch start, increment the index, and reset batch index
                self.epoch_start = None
                self.epoch_index += 1
                self.batch_index = 0
        elif message_name == pyromq.MESSAGES.BATCH_END:
            if self.batch_start is not None:
                self.add_entry('batch', time.time() - self.batch_start)
                # reset batch start and increment the index
                self.batch_start = None
                self.batch_index += 1

    def push_to_db(self):
        """
        Push the entries in self._db_entries into the database
        """
        if len(self._db_entries) == 0:
            return

        for _ in range(constants.NUM_SQL_RETRIES):
            try:
                pd.DataFrame(self._db_entries).to_sql("timing_events",
                                                      self.conn,
                                                      if_exists="append")
                self._db_entries = []
                return
            except (DatabaseError, sqlite3.OperationalError):
                time.sleep(1)

        pd.DataFrame(self._db_entries).to_sql("timing_events",
                                              self.conn,
                                              if_exists="append")
        self._db_entries = []


class DBSubscriber(pyromq.Subscriber):
    """
    The DBSubscriber subscribes to metric events and logs them into the
    target database. It can also save the parameters that were used in each
    training instantiation.

    .. seealso:: For how pandas interacts with sqlite3, please visit
        https://www.dataquest.io/blog/python-pandas-databases/
    """

    def __init__(self, experiment_uid, db_name):
        """
        :param experiment_uid: each training instantiation should have its own
            unique id so that it can be identified in the database
        :type experiment_uid: str

        :param db_name: The fully expanded path to the database
        :type db_name: str
        """
        self.experiment_uid = experiment_uid
        self.conn = sqlite3.connect(db_name)
        self.epoch = 0

    def training_events(self, message):
        """
        :param message: Training events are dictionaries of the format
            {'name': name_of_training_event, 'value': True or False}
        """
        if message['name'] == pyromq.MESSAGES.EPOCH_START:
            self.epoch += 1

        entries = {"experiment": [self.experiment_uid],
                   "event_name": [message['name']],
                   "event_status": [message['value']],
                   "timestamp": [datetime.now()]}

        for _ in range(constants.NUM_SQL_RETRIES):
            try:
                pd.DataFrame(entries).to_sql("training_events",
                                             self.conn,
                                             if_exists="append")
                return
            except (DatabaseError, sqlite3.OperationalError):
                time.sleep(1)

        pd.DataFrame(entries).to_sql("training_events",
                                     self.conn,
                                     if_exists="append")

    def metric_events(self, message):
        """
        Metric events are a bit more complex than training events simply because
            a metric might evaluate each of the classes separately (as is the case
            for the IndividualF1 measure).

            More specifically, there are two sitautions corresponding to two ways
            that metric events can emit information:

                1. message is of format {"name": metric_name, "value": numeric_value}
                2. message is of format {"name": metric_name, "value": a_dictionary}

            In the second case, the dictionary is a mapping from specific evaluations
            to numeric values:

                {specific_name: specific_value, ...}

        :param message: see the primary portion of this docstring.
        :type message: dict
        """
        metric_name, data_type = pyromq.unpack_name(message['name'])
        if isinstance(message['value'], dict):
            # this is the individual F1 case.  Custom handling here, plans to
            # streamline in next iteration of code condensing.
            entries = {"experiment": [], "metric_name": [], "data_type": [],
                       "value": [], "timestamp": [], "epoch": []}
            stamp = datetime.now()
            for specific_metric_name, value in message['value'].items():
                entries['experiment'].append(self.experiment_uid)
                entries['metric_name'].append(
                    metric_name + "." + specific_metric_name.lower())
                entries['data_type'].append(data_type)
                entries['value'].append(value)
                entries['timestamp'].append(stamp)
                entries['epoch'].append(self.epoch)
        else:
            entries = {"experiment": [self.experiment_uid],
                       "metric_name": [metric_name],
                       "data_type": [data_type],
                       "value": [message['value']],
                       "timestamp": [datetime.now()],
                       'epoch': [self.epoch]}

        for _ in range(constants.NUM_SQL_RETRIES):
            try:
                pd.DataFrame(entries).to_sql("metric_events",
                                             self.conn,
                                             if_exists="append")
                return
            except (DatabaseError, sqlite3.OperationalError):
                time.sleep(1)

        pd.DataFrame(entries).to_sql("metric_events",
                                     self.conn,
                                     if_exists="append")

    def store_metadata(self, metadata):
        """
        Store the metadata of an experiment into a table specifically for it

        Note: this will create a table with the columns that are present in
            metadata

        :param metadata: The message is assumed to be a non-nested dictionary
            of parameter configurations.
        :type metadata: dict of {str: numeric, ...}
        """
        logger.debug("Storing metadata")
        logger.debug(metadata)
        metadata = {k: [v] for k, v in metadata.items()}
        metadata['experiment'] = [self.experiment_uid]

        for i in range(constants.NUM_SQL_RETRIES):
            try:
                pd.DataFrame(metadata).to_sql("metadata",
                                              self.conn,
                                              if_exists="append")
                return
            except (DatabaseError, sqlite3.OperationalError):
                time.sleep(i+1)

        pd.DataFrame(metadata).to_sql("metadata",
                                      self.conn,
                                      if_exists="append")

    def store_posteriors(self, targets, posteriors, data_indices=None):
        targets = torchutils.numpy_from_torch(targets)
        posteriors = torchutils.numpy_from_torch(posteriors)
        if data_indices is not None:
            data_indices = torchutils.numpy_from_torch(data_indices)
        if len(targets.shape) == 2:
            targets = targets.argmax(axis=-1)
        to_sql_df = []
        for i in range(len(targets)):
            if data_indices is not None:
                data_index = data_indices[i]
            else:
                data_index = -1
            to_sql_df.append({'target_index': targets[i],
                              'posterior': json.dumps(posteriors[i].tolist()),
                              'experiment': self.experiment_uid,
                              'epoch': self.epoch,
                              'data_index': data_index})

        for _ in range(constants.NUM_SQL_RETRIES):
            try:
                pd.DataFrame(to_sql_df).to_sql("posteriors",
                                               self.conn,
                                               if_exists="append")
                return
            except (DatabaseError, sqlite3.OperationalError):
                time.sleep(1)

        pd.DataFrame(to_sql_df).to_sql("posteriors",
                                       self.conn,
                                       if_exists="append")


class TensorboardSubscriber(pyromq.Subscriber):
    """
    TODO:
        http://tensorboard-pytorch.readthedocs.io/en/latest/tensorboard.html
        self.writer.add_audio
        self.writer.add_text
        self.writer.add_histogram
        self.writer.add_image
        self.writer.add_embedding
    """

    def __init__(self):
        self.enabled = False
        try:
            import tensorboardX
            self.enabled = True
            self.writer = tensorboardX.SummaryWriter()
        except ImportError:
            logger.error(
                "please install https://github.com/lanpa/tensorboard-pytorch")
            # FIXME: Does it make sense to do other inits if the package is not present?
        self.epoch = 0
        self.metric_event_counter = 0
        self.epoch_averages = collections.defaultdict(list)

    def training_events(self, message):
        """
        :param message: Training events are dictionaries of the format
            {'name': name_of_training_event, 'value': True or False}
        """
        if not self.enabled:
            return

        if message['name'] == "epoch.start":
            self.epoch += 1

        if message['name'] == "epoch.end":

            for name, values in list(self.epoch_averages.items()):
                if len(values) == 0:
                    continue

                if isinstance(values[0], dict):
                    # collate the values
                    collated = {subname: [] for subname in values[0].keys()}
                    for value_dict in values:
                        for subname, subvalue in value_dict.items():
                            collated[subname].append(subvalue)
                    # average the collated values
                    collated = {subname: np.mean(subvalues) for
                                subname, subvalues in collated.items()}
                    # write them
                    self.writer.add_scalars(name, collated, self.epoch)
                else:
                    # easy case: just write the average
                    # noinspection PyTypeChecker
                    self.writer.add_scalar(name, np.mean(values), self.epoch)
                self.epoch_averages[name] = []

    def metric_events(self, message):
        """
        Metric events are a bit more complex than training events simply because
            a metric might evaluate each of the classes separately (as is the case
            for the IndividualF1 measure).

            More specifically, there are two sitautions corresponding to two ways
            that metric events can emit information:

                1. message is of format {"name": metric_name, "value": numeric_value}
                2. message is of format {"name": metric_name, "value": a_dictionary}

            In the second case, the dictionary is a mapping from specific evaluations
            to numeric values:

                {specific_name: specific_value, ...}

        :param message: see the primary portion of this docstring.
        :type message: dict
        """
        if not self.enabled:
            return

        metric_name, data_type = pyromq.unpack_name(message['name'])
        writer_name = '{}/{}'.format(metric_name, data_type)
        self.epoch_averages[writer_name].append(message['value'])
        writer_name += '/long'

        if isinstance(message['value'], dict):
            self.writer.add_scalars(writer_name, message['value'].items(),
                                    self.metric_event_counter)
        else:
            self.writer.add_scalar(writer_name, message['value'],
                                   self.metric_event_counter)

        self.metric_event_counter += 1


class LRScheduleSubscriber(pyromq.MessageSubscriber):
    def __init__(self, scheduler, target_metric_name=None):

        if isinstance(scheduler, (lr_scheduler.LambdaLR, lr_scheduler.StepLR,
                                  lr_scheduler.MultiStepLR,
                                  lr_scheduler.ExponentialLR)):
            self._pre_epoch = True
            self._requires_validation = False
        elif isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            if target_metric_name is None:
                raise ValueError("ReduceLROnPlateau requires a target metric "
                                 "name but `target_metric_name` is None")
            self._pre_epoch = False
            self._requires_validation = True
        else:
            raise RuntimeError("Unknown learning rate scheduler passed to "
                               f"{self.__class__.__name__}")

        self.scheduler = scheduler
        self._target_metric_name = target_metric_name
        self._cached_validation_values = []

    def training_events(self, message):
        message_name = message['name']
        if message_name == pyromq.MESSAGES.EPOCH_START and self._pre_epoch:
            self.scheduler.step()
            print("LOWERING LR")
        elif message_name == pyromq.MESSAGES.EPOCH_END and not self._pre_epoch:
            if len(self._cached_valiation_values) == 0:
                raise RuntimeError("Scheduler has not seen any validation "
                                   "values; perhaps training loops is badly "
                                   "formed and EPOCH_END is being signaled too "
                                   "early")
            val_value = np.mean(self._cached_valiation_values)
            self.scheduler.step(val_value)
            print("LOWERING LR")

    def metric_events(self, message):
        metric_name, data_type = pyromq.unpack_name(message['name'])
        if data_type == 'val' and metric_name == self._target_metric_name:
            self._cached_validation_values.append(message['value'])


class LogSubscriber(pyromq.Subscriber):
    def __init__(self, experiment_uid, log_file, to_file=True, to_console=True,
                 ignore_data_info_events=True, ignore_batch_events=True):

        self.experiment_uid = experiment_uid
        self.logger = logging.getLogger(experiment_uid)
        self.logger.setLevel(logging.DEBUG)
        self.ignore_data_info_events = ignore_data_info_events
        self.ignore_batch_events = ignore_batch_events
        self.epoch = 0

        formatter = logging.Formatter('[%(asctime)s] [%(name)s] ' +
                                      '[epoch%(epoch)s] [%(event_type)s] ' +
                                      '[%(message_name)s] [%(message)s]')

        if to_file:
            path_dir, _ = os.path.split(log_file)
            if not os.path.exists(path_dir):
                try:
                    os.makedirs(path_dir)
                except FileExistsError:
                    # was probably beaten to it by race condition
                    pass
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        if to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def training_events(self, message):
        if not isinstance(message, dict):
            message = {'name': message, 'value': 'success'}
        if self.ignore_batch_events and "batch" in message['name']:
            return
        if message['name'] == "epoch.start":
            self.epoch += 1

        self.generic_events(message, "training_events")

    def metric_events(self, message):
        self.generic_events(message, "metric_events")

    def controller_events(self, message):
        self.generic_events(message, "controller_events")

    def actuator_events(self, message):
        self.generic_events(message, "actuator_events")

    def info_events(self, message):
        if self.ignore_data_info_events and "data" in message['name']:
            return
        self.generic_events(message, "info_events")

    def generic_events(self, message, event_type):

        self.logger.info(message['value'],
                         extra={'event_type': event_type,
                                'epoch': self.epoch,
                                'message_name': message['name']})


class ModelCheckpoint(pyromq.PublishingSubscriber):
    def __init__(self, model, filepath, target_metric, target_data,
                 forced_interval=-1,
                 min_delta=0.01, name='model_checkpoint', mode='min',
                 channel_name=pyromq.CHANNELS.INFO_EVENTS,
                 broker=None):

        super(ModelCheckpoint, self).__init__(name=name,
                                              channel_name=channel_name,
                                              broker=broker)

        self.observed_events = []

        self.model = model
        self.filepath = filepath

        self.forced_interval = forced_interval
        self.epoch_index = 0

        self.target_metric = target_metric
        self.target_data = target_data
        self.min_delta = min_delta

        possible_metrics_to_mode = {"nll": "min",
                                    "js_divergence": "min",
                                    "macro_f1": "max",
                                    "micro_f1": "max",
                                    "accuracy": "max"}
        mode = possible_metrics_to_mode.get(target_metric, mode)
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater

        if self.monitor_op == np.less:
            self.min_delta *= -1
            self.best_value = 10 ** 10
        else:
            self.best_value = -10 ** 10

    def event_matches_target(self, message):
        metric_name, data_type = pyromq.unpack_name(message['name'])
        if self.target_metric != metric_name:
            return False
        if self.target_data != "*" and self.target_data != data_type:
            return False
        return True

    def publish(self, value, **kwargs):
        super(ModelCheckpoint, self).publish(self.name, value)

    def training_events(self, message):
        if message['name'] == pyromq.MESSAGES.EPOCH_END:
            value = np.mean(self.observed_events)
            self.observed_events = []

            if self.monitor_op(value - self.min_delta, self.best_value):
                self.publish(f"accepted! ({value:0.4f} beats "
                             f"{self.best_value:0.4f})")
                self.best_value = value
                self._save()
            else:
                self.publish(f"not accepted! ({value:0.4f} loses "
                             f"to {self.best_value:0.4f})")

            forced_condition = (self.forced_interval > 0 and
                                self.epoch_index > 0 and
                                self.epoch_index % self.forced_interval == 0)

            if forced_condition:
                self.publish(f"forced_interval;filepath={self.filepath}")
                left, right = os.path.splitext(self.filepath)
                self._save(f"{left}.{self.epoch_index}{right}")

            self.epoch_index += 1

    def _save(self, filepath=None):
        if filepath is None:
            filepath = self.filepath
        if hasattr(self.model, 'serialize_to_file'):
            self.model.serialize_to_file(filepath)
        elif hasattr(self.model, 'state_dict'):
            torch.save(self.model.state_dict(), filepath)
        else:
            raise Exception('unknown model type: {}'.format(repr(self.model)))

    def metric_events(self, message):
        if self.event_matches_target(message):
            self.observed_events.append(message['value'])


class EarlyStopping(pyromq.PublishingSubscriber):
    def __init__(self, target_metric, target_data, min_delta=0.1, patience=3,
                 mode='min', channel_name=pyromq.CHANNELS.INFO_EVENTS,
                 name='early_stopping', broker=None):
        super(EarlyStopping, self).__init__(name=name,
                                            channel_name=channel_name,
                                            broker=broker)
        self.observed_events = []

        self.target_metric = target_metric
        self.target_data = target_data
        self.min_delta = min_delta
        self.patience = patience

        self.num_waits = 0

        possible_metrics_to_mode = {"nll": "min",
                                    "js_divergence": "min",
                                    "macro_f1": "max",
                                    "micro_f1": "max",
                                    "accuracy": "max"}

        mode = possible_metrics_to_mode.get(target_metric, mode)
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater

        if self.monitor_op == np.less:
            self.min_delta *= -1
            self.best_value = 10 ** 10
        else:
            self.best_value = -10 ** 10

    def publish(self, value, **kwargs):
        super(EarlyStopping, self).publish(self.name, value)

    def event_matches_target(self, message):
        metric_name, data_type = pyromq.unpack_name(message['name'])
        if self.target_metric != "*" and self.target_metric != metric_name:
            return False
        if self.target_data != "*" and self.target_data != data_type:
            return False
        return True

    def training_events(self, message):
        if message['name'] == pyromq.MESSAGES.EPOCH_END:
            value = np.mean(self.observed_events)
            self.observed_events = []

            if self.monitor_op(value - self.min_delta, self.best_value):
                self.num_waits = 0
                self.best_value = value
                self.publish("{}|{}|{:0.4f}".format(self.target_metric,
                                                    self.target_data,
                                                    value))
            else:
                self.num_waits += 1
                self.publish("{}/{} patience used".format(self.num_waits,
                                                          self.patience))

    def metric_events(self, message):
        if self.event_matches_target(message):
            self.observed_events.append(message['value'])

    def should_stop(self):
        if 0 <= self.patience <= self.num_waits:
            self.publish("stopping criterion reached")
            return True
        return False

    def should_continue(self):
        return not self.should_stop()

# TODO Deprecate

TimeDBSubscriber = ChronometerSubscriber