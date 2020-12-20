import argparse

from trainer import Mutator
import os
import pickle
import torch
from dataset import get_dataset
import attr
from models import MLP, Controller


@attr.s
class Opts:
    dataset = attr.ib()
    num_task = attr.ib()
    num_class = attr.ib()
    class_per_task = attr.ib()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='mnist', help="sparc or cosql")
    parser.add_argument("--shuffle", type=bool, default=True, help="shuffle the train dataset")

    # controller opts
    parser.add_argument("--hidden", type=int, default=256, help="hidden size of rnn controller and embedding size")
    parser.add_argument("--n_layers", type=int, default=2, help="number of rnn layers")

    # controller model trains
    parser.add_argument("--controller_steps", type=int, default=100, help="train steps for controller")
    parser.add_argument("--controller_lr", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--controller_logging_step", type=int, default=20, help="log after x steps")

    # base model opts
    parser.add_argument("--base", type=str, default='mlp', help="base model name")
    parser.add_argument("--adapt", type=bool, default=False, help="base model adapt")

    # mlp model opts
    parser.add_argument("--mlp_size", type=int, default=512, help="mlp dim in and out")
    parser.add_argument("--mlp_linear", type=int, default=3, help="number of mlp layers")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout prob")

    # base model train
    parser.add_argument("--eval_steps", type=int, default=50, help="train n step and eval")
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=50, help="number of batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")

    #
    parser.add_argument("--with_cuda", type=bool, default=False, help="")

    args = parser.parse_args()
    if args.dataset == 'mnist':
        num_task = 5
        num_class = 10
        class_per_task = num_class // num_task

    elif args.dataset == 'cifar10':
        num_task = 5
        num_class = 10
        class_per_task = num_class // num_task
    else:
        raise NotImplemented

    opts = Opts(dataset=args.dataset, num_task=num_task, num_class=num_class, class_per_task=class_per_task)
    print("Loading {} Dataset".format(args.dataset))
    data = get_dataset(opts)

    mutator = Mutator(args, data, opts)
    mutator.run()
