"""
Pyromancy Message Queue classes and constants

There are three primary members:

    Broker
    Publisher
    Subscriber
"""
from collections import defaultdict

from pyromancy.utils.torchutils import numpy_from_torch
import torch


class CHANNELS:
    """
    names for the channels the pyromancy will send compute messages on
    """
    TRAINING_EVENTS = 'training_events'
    METRIC_EVENTS = 'metric_events'
    INFO_EVENTS = 'info_events'
    ACTUATOR_EVENTS = 'actuator_events'


class MESSAGES:
    """ names for the kinds of messages pyromancy will send """

    TRAINING_START = "training.start"
    TRAINING_END = "training.end"
    EPOCH_START = "epoch.start"
    EPOCH_END = "epoch.end"
    DATA_START = "data.start"
    DATA_END = "data.end"
    BATCH_START = "batch.start"
    BATCH_END = "batch.end"


def pack_name(publisher_name, data_name):
    """
    Construct a packed string used for naming pyromancy's compute messages.

    Args:
        publisher_name (str): the entity from which the message originates
            examples: 'accuracy', 'nll', 'model_checkpoint', 'early_stopping'
        data_name (str): the name of the data which the message originates
            examples: 'train', 'test', 'val'

    Returns:
        the packed message name (str)
    """
    return "{}|{}".format(publisher_name, data_name)


def unpack_name(packed_name):
    """
    Deconstruct a compact string used for naming pyromancy's compute messages.

    Args:
        packed_name (str):

    Returns:
        the unpacked message name (list / [str, str])
    """
    return packed_name.split("|")


class MessageBroker(object):
    """
    A MessageBroker is the main message coordinator in Pyromancy.

    Expectations of Broker:
        - the 'publish' function
        - the 'add_subscriber' function

    """

    def __init__(self, subscribers=None):
        """
        Args:
            subscribers (list): default None; an optional argument to Broker
        """
        if subscribers is None:
            subscribers = []
        self.subscribers = subscribers

    def publish(self, channel, message):
        """


        Args:
            channel (str): the channel name which is supposed to receive the
                message.  Downstream, it is currently being used by subscribers
                as the name of a function.

                The following is True: `assert hasattr(subscriber, channel)`

                Example: if `channel = 'training_events'`, then subscribers who
                    would like to subscribe to `training_events` should have a function
                    named `training_events`

            message (dict): a dictionary which has following keys:
                {'message_name': 'some_name',
                 'message_value': 'some_value'}
        """
        for subscriber in self.subscribers:
            subscriber.receive(message, channel)

    def add_subscriber(self, subscriber):
        """
        Add a subscriber to listen to the messages routed through this broker.

        Args:
            subscriber (pyromq.Subscriber): the subscriber to add
        """
        self.subscribers.append(subscriber)


class MessagePublisher(object):
    """
    The second core member of pyromq which sends out the compute messages.

    A Publisher, at the minimum, has a reference to a pyromq.Broker the publish
    method which packages a message_name and message_value.  This is the class
    which creates the expectation of what the messages that Broker receives will
    look.
    """

    def __init__(self, name, channel_name="default", broker=None):
        """
        Args:
            name (str): the name which will be attached to all messages
            channel_name (str): the channel upon which this publisher should
                send its messages on.
                TODO: consider this downside; Publishers can only publish on a
                single channel_name.
            broker (pyromq.Broker or None): optionally pass in the Broker.
                Without the Broker, the Publisher will not emit any of its
                messages. This is useful for stubbing and is therefore optional.
        """
        self.name = name
        self.channel_name = channel_name
        self.broker = broker

    def publish(self, message_name, message_value):
        """
        Args:
            message_name (str): the name of the message
                Examples: 'accuracy', 'nll', ...
            message_value (*): the message_value can be any type.. whatever it
                emits though should be expected by the subscribers on the
                channel_name of this Publisher
        """
        if self.broker is not None:
            message = {'name': message_name, 'value': message_value}
            self.broker.publish(self.channel_name, message)


class TrainingEventPublisher(MessagePublisher):
    """
    A wrapper around the training event messages such as EPOCH and BATCH messages

    These can be used to trigger periodic things, such as model checkpoint at
    the end of an epoch, a batch counter/stopwatch, etc.
    """

    def __init__(self, broker=None):
        super(TrainingEventPublisher, self).__init__(
            name='training_event_publisher',
            channel_name=CHANNELS.TRAINING_EVENTS,
            broker=broker)

    def training_start(self, message_value='success'):
        self.publish(MESSAGES.TRAINING_START, message_value)

    def training_end(self, message_value='success'):
        self.publish(MESSAGES.TRAINING_END, message_value)

    def epoch_start(self, message_value='success'):
        self.publish(MESSAGES.EPOCH_START, message_value)

    def epoch_end(self, message_value='success'):
        self.publish(MESSAGES.EPOCH_END, message_value)

    def data_start(self, message_value='success'):
        self.publish(MESSAGES.DATA_START, message_value)

    def data_end(self, message_value='success'):
        self.publish(MESSAGES.DATA_END, message_value)

    def batch_start(self, message_value='success'):
        self.publish(MESSAGES.BATCH_START, message_value)

    def batch_end(self, message_value='success'):
        self.publish(MESSAGES.BATCH_END, message_value)


class ComputePublisher(MessagePublisher):
    """
    A Publisher subclass which serves as the Base Class for all Metrics & Losses

    TODO: Make this an abc derived class
    """

    def compute(self, in_dict, out_dict):
        """
        The compute method assumes a compute stream is ongoing, with the
            in_dict being the incoming data, and the out_dict being the outgoing
            data.

        Example:
            model = SomeModel()
            in_dict = { 'x_data': some_tensor,
                        'y_target': some_tensor}
            out_data = {'y_predictions': model(in_dict['x_data'])}

        A ComputePublisher would then compute the accuracy given targets and predictions
        Another ComputePublisher could then compute the Neg. Log Likelihood loss

        Args:
            in_dict (dict): An input dictionary to the compute stream; it should
                be a mapping between tensor names and tensors
            out_dict (dict): An output dictionary to the compute stream; it should
                be a mapping between tensor names and tensors
        """
        raise NotImplemented

    def __call__(self, in_dict, out_dict, data_type):
        """
        Overriding the data object model method __call__ in order to give a
        functional use out of the ComputePublisher

        - Overall, it uses the self.compute method to compute a value.
        - If that value is a pytorch value, it will also create a python data
        type (probably numpy type) and emit that to the publisher
        - It will emit the object that was returned from compute
        """
        out = {'python_value': self.compute(in_dict, out_dict)}

        if isinstance(out['python_value'], torch.autograd.Variable):
            out['pytorch_value'] = out['python_value']
            # TODO: right now 0 is hard coded because of how pytorch yields
            #   singletons, such as in accuracy computations. See if this can be
            #   made better
            out['python_value'] = numpy_from_torch(out['python_value'])[0]

        message_name = pack_name(publisher_name=self.name, data_name=data_type)

        self.publish(message_name, out['python_value'])

        return out


class PublisherGroup(MessagePublisher):
    """
    Coordinates a group of publishers and their data target.

    This class is used as the base class because two different classes follow
    this pattern:
        - LossGroup: losses have additional functionality required for stepping
            down the gradient
        - MetricGroup: metrics just need to compute and move on

    """

    def __init__(self, *args, **kwargs):
        """
        Although the arguments are being passed through, they should be
            name, channel_name, broker

        This __init__ also creates self.members_by_target, which is a defaultdict
        that assigns data targets to lists of publishers for that data target.
        Allows for specifying train, test, or val.
        """
        super(PublisherGroup, self).__init__(*args, **kwargs)
        self.members_by_target = defaultdict(list)

    def add(self, member_object, data_target="*"):
        """
        Args:
            member_object (pyromq.Publisher): the object which will be computing
                with the rest of this group.
                For example, could be a Loss or a Metric
            data_target (str):
                the name of the data type. For example, 'train', 'val', or 'test'
                Passing in an asterisk ('*') will fire on all data types
        """
        member_object.channel_name = self.channel_name
        member_object.broker = self.broker
        self.members_by_target[data_target].append(member_object)


class MessageSubscriber(object):
    """
    The final component of the pyromq trifecta: pyromq.Subscriber

    A Subscriber represents the listeners that just do stuff with the compute
    stream and do not emit messages about it. For example, they could log messages
    to a file, to a database, or Tensorboard.  They could also save the model, etc.
    """

    def receive(self, message, channel):
        """
        Args:
            message (dict): The dictionary of
                {'message_name': message_name,
                 'message_value': message_value}
                which was published by a Publisher
            channel (str): the name of the channel upon which the message was
                emitted.  At the moment, all subscribers hear all channels.
                The filter lies in the object having the proper method to
                listen to the message.
        """
        if hasattr(self, channel):
            channel_func = getattr(self, channel)
            channel_func(message)

    def training_events(self, message):
        pass

    def metric_events(self, message):
        pass

    def actuator_events(self, message):
        pass


class PublishingSubscriber(MessagePublisher, MessageSubscriber):
    """
    a mixin of both the Publisher and Subscriber.

    see pyromancy.subscribers for some examples.
    """


# FOR BACKWARD COMPATABILITY
# TODO: deprecate
channels = CHANNELS
messages = MESSAGES
Broker = MessageBroker
Publisher = MessagePublisher
Subscriber = MessageSubscriber
