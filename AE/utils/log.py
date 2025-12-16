from matplotlib  import pyplot as plt


def log_epoch(writer, modus, epoch, loss, cr, bpppc, mse, psnr, sa, data_org, data_rec):
    # log losses / metrics
    writer.add_scalar(modus + "/loss", loss, epoch)
    writer.add_scalar(modus + "/cr", cr, epoch)
    writer.add_scalar(modus + "/bpppc", bpppc, epoch)
    writer.add_scalar(modus + "/mse", mse, epoch)
    writer.add_scalar(modus + "/psnr", psnr, epoch)
    writer.add_scalar(modus + "/sa", sa, epoch)

    # log spectral signature
    fig = plt.figure()

    bands = [i for i in range(1, 203)]

    tensor_size = data_org.size()
    index = tensor_size[2] // 2

    data_org_spec = data_org[-1, :, index, index].detach().cpu()
    data_rec_spec = data_rec[-1, :, index, index].detach().cpu()

    plt.plot(bands, data_org_spec, linewidth=2.0)
    plt.plot(bands, data_rec_spec, linewidth=2.0)
    
    plt.xlim(1, 202)
    plt.ylim(0, 1) # TODO

    plt.grid(which='both')
    plt.xlabel('Band')
    plt.ylabel('Value')
    plt.legend(['org', 'rec'], loc='upper right', fontsize='xx-small')

    writer.add_figure(modus + "/_spec", fig, epoch)

    plt.close(fig)

    # log images
    idx_c = [44, 29, 11]  # r, g, b channels
    # log original image batch
    writer.add_images(f"{modus}/_org", data_org[:, idx_c, :, :]/10_000, epoch, dataformats='NCHW')
    # log reconstructed image batch
    writer.add_images(f"{modus}/_rec", data_rec[:, idx_c, :, :]/10_000, epoch, dataformats='NCHW')


def log_hparams(
        writer,
        args,
        metric_dict={
            "test/cr": 0,
            "test/bpppc": 0,
            "test/mse": 0,
            "test/psnr": 0,
            "test/sa": 0,
            "test/enc_time": 0,
            "test/dec_time": 0,
        },
        ):
    hparam_dict = {
        "model": args.model,
        "learning-rate": args.learning_rate,
        "device": args.devices[0],
        "dataset": args.dataset,
        "mode": args.mode,
        "transform": args.transform,
        "train-batch-size": args.train_batch_size,
        "val-batch-size": args.val_batch_size,
        "num-workers": args.num_workers,
        "dataset": args.dataset,
        "num-channels": args.num_channels,
        "model": args.model,
        "loss": args.loss,
        "epochs": args.epochs,
        "learning-rate": args.learning_rate,
        "seed": args.seed,
        "optimizer": args.optimizer,
    }

    writer.add_hparams(hparam_dict, metric_dict, run_name=".")
