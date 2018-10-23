import argparse
import logging

from pyromancy.experiment import Trial, RQExperiment

# noinspection PyUnresolvedReferences
from lib import fail, run_once

logging.getLogger().setLevel(logging.INFO)


trial1 = Trial('Trial 1', val1='foo', val2=3.1415, val3=[1, 2, 3])
trial2 = Trial('Trial 2', val1='bar', val2=3.1415*2, val3=[4, 5, 6])
trials = [trial1, trial2]
cli_args = argparse.Namespace(val1=1.618)

experiment_1 = RQExperiment('Experiment 1', trials)
experiment_1.run(cli_args, run_once, poll=True, single_thread=False, requeue=True)

experiment_2 = RQExperiment('Experiment 2', trials)
experiment_2.run(cli_args, fail, poll=True, single_thread=False, requeue=True)
