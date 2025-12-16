import os

from torch import nn

from utils import time


def get_save_dir(args):
    if args.checkpoint is None:
        save_dir = f"{args.save_dir}{time.get_date_time()}-{get_run_name(args)}/"
    else:
        save_dir = f"{os.path.dirname(args.checkpoint)}/"
    return save_dir


def get_run_name(args):
    if args.transform is None:
        save_dir = f"{args.mode}-{args.model}-{args.loss}-{args.learning_rate}"
    else:
        save_dir = f"{args.mode}_{args.transform}-{args.model}-{args.loss}-{args.learning_rate}"
    return save_dir


class AverageMeter:
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        val = val.data
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)
