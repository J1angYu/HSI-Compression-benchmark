import shutil
import torch


def load_checkpoint_eval(checkpoint_path, net):
    print("Loading", checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()


def load_checkpoint_train(checkpoint_path, net, optimizer, device):
    print("Loading", checkpoint_path, "to continue training.")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    last_epoch = checkpoint["epoch"] + 1
    net.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return last_epoch


def save_checkpoint(state, is_best, save_dir="./results/", filename="last.pth.tar"):
    save_path_last = f"{save_dir}{filename}"
    torch.save(state, save_path_last)
    if is_best:
        save_path_best = f"{save_dir}best.pth.tar"
        shutil.copyfile(save_path_last, save_path_best)


def strip_checkpoint(checkpoint_path, save_dir="./results/", filename="final.pth.tar"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {
        "state_dict": checkpoint["state_dict"]
    }
    torch.save(state_dict, f"{save_dir}{filename}")
