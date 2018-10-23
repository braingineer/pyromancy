from __future__ import print_function

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from tqdm import tqdm

from pyromancy import pyromq
from pyromancy.losses import LossGroup, NegativeLogLikelihood
from pyromancy.metrics import MetricGroup, Accuracy
from pyromancy.subscribers import LogSubscriber
from pyromancy.subscribers import ModelCheckpoint


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

    parser.add_argument('--experiment-name', required=True)
    parser.add_argument("--experimentdb", default=None)

    parser.add_argument('--log-to-console', default=False, action='store_true')

    parser.add_argument('--checkpoint-min-delta', default=0.01, type=float)
    parser.add_argument('--reload-checkpoint', default=False, action='store_true')

    args = parser.parse_args()

    if args.experimentdb is None:
        args.experimentdb = args.experiment_name + '.db'

    return args


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


# noinspection PyCallingNonCallable,PyCallingNonCallable
def run_once(args, train_loader, test_loader):
    broker = pyromq.Broker()

    model = Net()
    if args.cuda:
        model.cuda()

    training_events = pyromq.TrainingEventPublisher(broker=broker)

    broker.add_subscriber(LogSubscriber(experiment_uid=args.experiment_name,
                                        log_file=os.path.join('logs', args.experiment_name),
                                        to_console=args.log_to_console))

    model_filename = args.experiment_name + '.state'

    broker.add_subscriber(ModelCheckpoint(model, model_filename,
                                          target_metric='nll',
                                          target_data='val',
                                          broker=broker,
                                          min_delta=args.checkpoint_min_delta))

    if args.reload_checkpoint and os.path.exists(model_filename):
        # TODO make this an info event
        print("Reloading checkpoint!")
        model.load_state_dict(torch.load(model_filename))

    opt = torch.optim.SGD(params=model.parameters(),
                          lr=args.lr,
                          weight_decay=args.weight_decay,
                          momentum=args.momentum)

    losses = LossGroup(optimizer=opt,
                       grad_clip_norm=args.grad_clip_norm,
                       name='losses',
                       channel_name=pyromq.channels.METRIC_EVENTS,
                       broker=broker)

    losses.add(NegativeLogLikelihood(name='nll',
                                     target_name='y_target',
                                     output_name='y_pred'),
               data_target='train')

    # Metrics

    metrics = MetricGroup(name='metrics',
                          channel_name=pyromq.channels.METRIC_EVENTS,
                          broker=broker)

    metrics.add(Accuracy(name='acc',
                         target_name='y_target',
                         output_name='y_pred'),
                data_target='*')

    metrics.add(NegativeLogLikelihood(name='nll',
                                      target_name='y_target',
                                      output_name='y_pred'),
                data_target='val')

    training_events.training_start()

    for _ in tqdm(range(args.epochs), total=args.epochs):
        training_events.epoch_start()

        model.train(True)

        for data, target in train_loader:
            # From the original example
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            # put the incoming batch data into a dictionary
            batch_dict = {'x_data': data, 'y_target': target}

            # Training Event
            training_events.batch_start()

            # Get model outputs
            predictions = {'y_pred': model(batch_dict['x_data'])}

            # Compute Metrics
            metrics.compute(in_dict=batch_dict, out_dict=predictions,
                            data_type='train')

            # Compute Losses
            losses.compute(in_dict=batch_dict, out_dict=predictions,
                           data_type='train')
            losses.step()

            # Training Event
            training_events.batch_end()

        model.train(False)

        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            batch_dict = {'x_data': data, 'y_target': target}

            # Training Event
            training_events.batch_start()

            predictions = {'y_pred': model(batch_dict['x_data'])}

            metrics.compute(in_dict=batch_dict,
                            out_dict=predictions,
                            data_type='val')

            training_events.batch_end()

        training_events.epoch_end()


def main():
    args = parse_args()
    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    dataload_kwargs = {}
    if args.cuda:
        dataload_kwargs = {'num_workers': 1, 'pin_memory': True}

    train_dataset = datasets.MNIST('../data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))])
                                   )
    # noinspection PyUnresolvedReferences
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True, **dataload_kwargs)

    test_dataset = datasets.MNIST('../data', train=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])
                                  )
    # noinspection PyUnresolvedReferences
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True, **dataload_kwargs)

    run_once(args, train_loader, test_loader)


if __name__ == "__main__":
    main()
