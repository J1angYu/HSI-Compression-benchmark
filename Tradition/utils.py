import torch
from pytorch_msssim import SSIM
import torch
import scipy.io
import numpy as np

################## dataset ##################
def load_hsi_data(file_path, device='cuda', dtype=torch.float32):
    """
    Load hyperspectral image data from a .mat file, normalize, and move to the specified device.

    Args:
        file_path (str): Path to the .mat data file.
        device (str): Device name ('cuda' or 'cpu').
        dtype (torch.dtype): Data type for the tensor.

    Returns:
        torch.Tensor: Hyperspectral image tensor of shape [C, H, W].
    """
    img_data = scipy.io.loadmat(file_path)
    img = next(v for k, v in img_data.items()
                if not k.startswith('__') and isinstance(v, np.ndarray) and len(v.shape) == 3)
    img = torch.from_numpy(img.astype(np.float32))
    img = img.permute(2, 0, 1)  # Convert to [C, H, W]

    # Normalize to [0, 1]
    img = (img - img.min()) / (img.max() - img.min())

    # Move to specified device and dtype
    img = img.to(device, dtype)

    return img


################## metrics ##################
def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 10 * torch.log10(1.0 / mse).item()

def compute_ssim(img1, img2):
    ssim_calculator = SSIM(data_range=1.0, channel=img1.shape[0])
    return ssim_calculator(img1.unsqueeze(0), img2.unsqueeze(0)).item()

def compute_sam(img1, img2):
    # C, H, W -> H*W, C
    img1_flat = img1.reshape(img1.shape[0], -1).permute(1, 0)
    img2_flat = img2.reshape(img2.shape[0], -1).permute(1, 0)
    
    cos_sim = torch.nn.functional.cosine_similarity(img1_flat, img2_flat, dim=1)
    
    sam = torch.mean(torch.acos(cos_sim)).item() 
    
    return sam