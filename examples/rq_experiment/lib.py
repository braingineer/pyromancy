
from torch import nn

from pyromancy import pyromq
import time


def run_once(args):
    # FIXME: Whatever this is supposed to be doing it is not.
    # 1. Fix this
    # 2. see if you should be using print or logging
    broker = pyromq.Broker()
    training_events = pyromq.TrainingEventPublisher(broker=broker)
    print(args)
    conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, dilation=1)
    print(conv)
    print("DONE!")
    time.sleep(5)
    return "result stub"


def fail(args):
    return 1/0
