"""
# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import argparse
import random
import sys
import time
import torch
import torchvision

from torch import optim
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.hyspecnet11k import HySpecNet11k

from losses import losses
from metrics import metrics
from models import models

from utils import checkpoint
from utils import log
from utils import util


def train_epoch(model, criterion, train_dataloader, optimizer, epoch, clip_max_norm, writer):
    model.train()
    device = next(model.parameters()).device

    bpppc = model.bpppc
    cr = model.compression_ratio

    mse_metric = metrics["mse"]()
    psnr_metric = metrics["psnr"]()
    sa_metric = metrics["sa"]()

    loss_meter = util.AverageMeter()
    mse_meter = util.AverageMeter()
    psnr_meter = util.AverageMeter()
    sa_meter = util.AverageMeter()

    loop = tqdm(train_dataloader, leave=True)
    loop.set_description(f"Epoch {epoch} Training  ")
    for data_org in loop:
        data_org = data_org.to(device)

        optimizer.zero_grad()

        data_rec = model(data_org)

        out_criterion = criterion(data_org, data_rec)

        out_criterion.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        # compute metrics
        loss = out_criterion
        mse = mse_metric(data_org, data_rec)
        psnr = psnr_metric(data_org, data_rec)
        sa = sa_metric(data_org, data_rec)

        # update metric averages
        loss_meter.update(loss)
        mse_meter.update(mse)
        psnr_meter.update(psnr)
        sa_meter.update(sa)

        # update progress bar to show results of current batch
        loop.set_postfix(
            _loss=loss.item(),
            cr=cr,
            bpppc=bpppc,
            mse=mse.item(),
            psnr=psnr.item(),
            sa=sa.item(),
        )

    # get average metrics over whole epoch
    loss_avg = loss_meter.avg.item()
    mse_avg = mse_meter.avg.item()
    psnr_avg = psnr_meter.avg.item()
    sa_avg = sa_meter.avg.item()

    # update progress bar to show results of whole training epoch
    loop.set_postfix(
        _loss=loss_avg,
        cr=cr,
        bpppc=bpppc,
        mse=mse_avg,
        psnr=psnr_avg,
        sa=sa_avg,
    )
    loop.update()
    loop.refresh()

    # log to tensorboard
    log.log_epoch(writer, "train", epoch, loss_avg, cr, bpppc, mse_avg, psnr_avg, sa_avg, data_org, data_rec)


def val_epoch(epoch, val_dataloader, model, criterion, writer):
    model.eval()
    device = next(model.parameters()).device

    bpppc = model.bpppc
    cr = model.compression_ratio

    mse_metric = metrics["mse"]()
    psnr_metric = metrics["psnr"]()
    sa_metric = metrics["sa"]()

    loss_meter = util.AverageMeter()
    mse_meter = util.AverageMeter()
    psnr_meter = util.AverageMeter()
    sa_meter = util.AverageMeter()

    with torch.no_grad():
        loop = tqdm(val_dataloader, leave=True)
        loop.set_description(f"Epoch {epoch} Validation")
        for data_org in loop:
            data_org = data_org.to(device)

            data_rec = model(data_org)

            out_criterion = criterion(data_org, data_rec)

            # compute metrics
            loss = out_criterion
            mse = mse_metric(data_org, data_rec)
            psnr = psnr_metric(data_org, data_rec)
            sa = sa_metric(data_org, data_rec)

            # update metric averages
            loss_meter.update(loss)
            mse_meter.update(mse)
            psnr_meter.update(psnr)
            sa_meter.update(sa)

            # update progress bar to show results of current batch
            loop.set_postfix(
                _loss=loss.item(),
                cr=cr,
                bpppc=bpppc,
                mse=mse.item(),
                psnr=psnr.item(),
                sa=sa.item(),
            )

        # get average metrics over whole validation set
        loss_avg = loss_meter.avg.item()
        mse_avg = mse_meter.avg.item()
        psnr_avg = psnr_meter.avg.item()
        sa_avg = sa_meter.avg.item()

        # update progress bar to show results of whole validation set
        loop.set_postfix(
            _loss=loss_avg,
            cr=cr,
            bpppc=bpppc,
            mse=mse_avg,
            psnr=psnr_avg,
            sa=sa_avg,
        )
        loop.update()
        loop.refresh()

        # log to tensorboard
        log.log_epoch(writer, "val", epoch, loss_avg, cr, bpppc, mse_avg, psnr_avg, sa_avg, data_org, data_rec)

    return loss_avg


def test_final(model, test_dataloader):
    model.eval()
    device = next(model.parameters()).device

    bpppc = model.bpppc
    cr = model.compression_ratio

    mse_metric = metrics["mse"]()
    psnr_metric = metrics["psnr"]()
    sa_metric = metrics["sa"]()

    mse_meter = util.AverageMeter()
    psnr_meter = util.AverageMeter()
    sa_meter = util.AverageMeter()
    enc_time_meter = util.AverageMeter()
    dec_time_meter = util.AverageMeter()

    with torch.no_grad():
        loop = tqdm(test_dataloader, leave=True)
        loop.set_description(f"Testing")

        for data_org in loop:
            data_org = data_org.to(device)

            start = time.time()
            data_lat = model.compress(data_org)
            enc_time = time.time() - start
            enc_time = torch.tensor([enc_time])

            start = time.time()
            data_rec = model.decompress(data_lat)
            dec_time = time.time() - start
            dec_time = torch.Tensor([dec_time])

            # compute metrics
            mse = mse_metric(data_org, data_rec)
            psnr = psnr_metric(data_org, data_rec)
            sa = sa_metric(data_org, data_rec)

            # update metric averages
            mse_meter.update(mse)
            psnr_meter.update(psnr)
            sa_meter.update(sa)
            enc_time_meter.update(enc_time)
            dec_time_meter.update(dec_time)

            # update progress bar to show results of current batch
            loop.set_postfix(
                cr=cr,
                bpppc=bpppc,
                mse=mse.item(),
                psnr=psnr.item(),
                sa=sa.item(),
                enc=enc_time.item(),
                dec=dec_time.item(),
            )

        # get average metrics over whole validation set
        mse_avg = mse_meter.avg.item()
        psnr_avg = psnr_meter.avg.item()
        sa_avg = sa_meter.avg.item()
        enc_time_avg = enc_time_meter.avg.item()
        dec_time_avg = dec_time_meter.avg.item()

        # update progress bar to show results of whole validation set
        loop.set_postfix(
            cr=cr,
            bpppc=bpppc,
            mse=mse_avg,
            psnr=psnr_avg,
            sa=sa_avg,
            enc=enc_time_avg,
            dec=dec_time_avg,
        )
        loop.update()
        loop.refresh()

    return {
        "test/cr": cr,
        "test/bpppc": bpppc,
        "test/mse": mse_avg,
        "test/psnr": psnr_avg,
        "test/sa": sa_avg,
        "test/enc_time": enc_time_avg,
        "test/dec_time": dec_time_avg,
    }


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    save_dir = util.get_save_dir(args)
    writer = SummaryWriter(log_dir=save_dir)

    log.log_hparams(writer, args)

    transform = None
    if args.transform is not None:
        if args.transform == "centercrop_16x16":
            transform = torchvision.transforms.CenterCrop(16)
            random_subsample_factor = None
        elif args.transform == "randomcrop_16x16":
            transform = torchvision.transforms.RandomCrop(16)
            random_subsample_factor = None
        elif args.transform == "random_1x1":
            transform = None
            random_subsample_factor = 128
        elif args.transform == "random_2x2":
            transform = None
            random_subsample_factor = 64
        elif args.transform == "random_4x4":
            transform = None
            random_subsample_factor = 32
        elif args.transform == "random_8x8":
            transform = None
            random_subsample_factor = 16
        elif args.transform == "random_16x16":
            transform = None
            random_subsample_factor = 8
        elif args.transform == "random_32x32":
            transform = None
            random_subsample_factor = 4
        elif args.transform == "random_64x64":
            transform = None
            random_subsample_factor = 2
    else:
        transform = None
        random_subsample_factor = None

    train_dataset = HySpecNet11k(args.dataset, split="train", mode=args.mode, transform=transform, random_subsample_factor=random_subsample_factor)
    val_dataset = HySpecNet11k(args.dataset, split="val", mode=args.mode, transform=transform, random_subsample_factor=random_subsample_factor)
    device = f"cuda:{args.devices[0]}" if args.devices[0] != "cpu" and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False
    )

    model = models[args.model](src_channels=args.num_channels)
    model = model.to(device)

    if args.devices != "cpu" and len(args.devices) > 1 and torch.cuda.device_count() > 1:
        model = util.CustomDataParallel(model, device_ids=args.devices)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    criterion = losses[args.loss]()

    last_epoch = 0
    # load from previous checkpoint
    if args.checkpoint:
        last_epoch = checkpoint.load_checkpoint_train(args.checkpoint, model, optimizer, device)

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        train_epoch(
            model,
            criterion,
            train_dataloader,
            optimizer,
            epoch,
            args.clip_max_norm,
            writer,
        )

        loss = val_epoch(epoch, val_dataloader, model, criterion, writer)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "loss": loss,
            "optimizer": optimizer.state_dict(),
        }
        checkpoint.save_checkpoint(state, is_best, save_dir=save_dir)

    # save weights of the best epoch without additional information
    checkpoint.strip_checkpoint(f"{save_dir}best.pth.tar", save_dir=save_dir)

    # evaluate best model on test set
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)

    test_dataset = HySpecNet11k(args.dataset, split="test", mode=args.mode, transform=None)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False,
        drop_last=False
    )

    checkpoint.load_checkpoint_eval(f"{save_dir}final.pth.tar", model)

    metric_dict = test_final(model, test_dataloader)

    log.log_hparams(writer, args, metric_dict)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Train script.")
    parser.add_argument(
        "--devices",
        # type=int,
        default=[0],
        nargs="+",
        help="Devices to use (default: %(default)s), e.g. cpu or 0 or 0,2,5,7 for multiple"
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=2,
        help="Training batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=2,
        help="Validation batch size (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Data loaders threads (default: %(default)s)",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="./datasets/hyspecnet-11k/",
        help="Path to dataset (default: %(default)s)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="easy",
        choices=["easy", "hard", "mini"],
        help="Dataset split mode (default: %(default)s)"
    )
    parser.add_argument(
        "--transform",
        type=str,
        default=None,
        choices=[
            None,
            "centercrop_16x16", "randomcrop_16x16",
            "random_1x1", "random_2x2", "random_4x4", "random_8x8", "random_16x16", "random_32x32", "random_64x64"
        ],
        help="Dataset transformation (default: %(default)s)"
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=202,
        help="Number of data channels, (default: %(default)s)"
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="hycot_cr4",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "--loss",
        default="mse",
        choices=losses.keys(),
        type=str,
        help="Loss (default: %(default)s)",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        default=2000,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-3,
        type=float,
        help="Learning rate (default: %(default)s)",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="./results/trains/",
        help="Didata_rectory to save results (default: %(default)s)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10587,
        help="Set random seed for reproducibility (default: %(default)s)"
    )
    parser.add_argument(
        "--clip-max-norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training"
    )

    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    main(sys.argv[1:])
