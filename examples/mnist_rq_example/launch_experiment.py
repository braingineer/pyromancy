import argparse
import logging
import os

from pyromancy.experiment import RQExperiment, Trial
import torch

# noinspection PyUnresolvedReferences
from lib import run_once

logging.getLogger().setLevel(logging.INFO)


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')

    parser.add_argument('--weight-decay', default=1e-4, type=float)

    parser.add_argument('--grad-clip-norm', default=10.0, type=float)

    parser.add_argument('--disable-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Name the experiment

    parser.add_argument('--experiment-name', default='MNIST-V1')

    parser.add_argument("--experimentdb", default=None)

    parser.add_argument('--log-to-console', default=False, action='store_true')

    parser.add_argument('--checkpoint-min-delta', default=0.01, type=float)
    parser.add_argument('--reload-checkpoint', default=False, action='store_true')

    parser.add_argument('--cache-dir', default=None, type=str)

    args = parser.parse_args()

    if args.cache_dir is None:
        args.cache_dir = 'cache/'+args.experiment_name

    if not os.path.exists(args.cache_dir):
        # noinspection PyTypeChecker
        os.makedirs(args.cache_dir)

    if args.experimentdb is None:
        args.experimentdb = args.experiment_name + '.db'

    return args


def main():
    args = parse_args()
    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    trials = []
    for lr in [0.1, 0.01, 0.001]:
        for weight_decay in [1e-1, 1e-2, 1e-3, 1e-4]:
            trials.append(Trial(f'mnist.{len(trials)}.lr{lr}.decay{weight_decay}',
                                lr=lr,
                                weight_decay=weight_decay))

    experiment = RQExperiment(args.experiment_name, trials)
    experiment.run(args, run_once, poll=True)


if __name__ == '__main__':
    main()
