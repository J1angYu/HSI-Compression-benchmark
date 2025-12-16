import torch
from pytorch_msssim import SSIM
import torch
import scipy.io
import numpy as np
from copy import deepcopy

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

def to_coordinates_and_features(img):
    """
    Convert an HSI image tensor to coordinate-feature pairs for implicit neural representations.

    Args:
        img (torch.Tensor): Hyperspectral image tensor of shape (channels, height, width)

    Returns:
        tuple: (coordinates, features)
            coordinates (torch.Tensor): Normalized coordinates, shape (height*width, 2)
            features (torch.Tensor): Feature tensor, shape (height*width, channels)
    """
    # Generate 2D pixel coordinate indices
    C, H, W = img.shape
    coordinates = torch.ones((H, W)).nonzero(as_tuple=False).float()  # shape: [N, 2] (N = H * W)
    coordinates = torch.flip(coordinates, [1])  # flip to [x, y] order

    # Normalize coordinates to [-1, 1]
    coordinates[:, 0] = (coordinates[:, 0] / (W - 1) - 0.5) * 2.0
    coordinates[:, 1] = (coordinates[:, 1] / (H - 1) - 0.5) * 2.0

    # Reshape features to [N, C]
    features = img.reshape(C, -1).T

    return coordinates, features


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


################## quantize ##################
def quantize_tensor(tensor, bits, axis=-1):
    """
    Performs affine quantization on a tensor.

    Args:
        tensor (torch.Tensor): The input tensor to quantize.
        bits (int): The number of bits for quantization.
        axis (int): The axis along which to compute min/max for quantization.
                    -1 means per-tensor quantization.
                    0 means per-channel quantization).
                    Other axes can be specified if needed.

    Returns:
        tuple:
            - quant_info (dict): A dictionary containing:
                - 'min_val' (torch.Tensor): The minimum value(s) used for scaling.
                - 'scale' (torch.Tensor): The scale factor(s) used.
                - 'quant_val' (torch.Tensor): The quantized integer values.
            - dequantized_tensor (torch.Tensor): The tensor after de-quantization (for evaluation).
    """
    if bits == -1: # No quantization
        quant_info = {
            'min_val': torch.tensor(0.0, device=tensor.device),
            'scale': torch.tensor(1.0, device=tensor.device),
            'quant_val': tensor.clone() # Store original values if not quantizing
        }
        return quant_info, tensor.clone()

    if axis == -1: # Per-tensor quantization
        min_val = tensor.min()
        max_val = tensor.max()
    else: # Per-axis quantization
        min_val = tensor.amin(dim=axis, keepdim=True)
        max_val = tensor.amax(dim=axis, keepdim=True)

    # Calculate scale
    # Ensure scale is not zero, handle case where min_val == max_val
    scale = (max_val - min_val) / (2**bits - 1)
    scale = torch.where(scale == 0, torch.tensor(1.0, device=scale.device, dtype=scale.dtype), scale)

    # Quantize
    quant_val = ((tensor - min_val) / scale).round()
    # 根据位数选择合适的整数类型
    if bits <= 8:
        dtype = torch.uint8
    elif bits <= 15:
        dtype = torch.int16
    elif bits <= 31:
        dtype = torch.int32
    else:
        raise ValueError("Unsupported number of bits for quantization.")
    quant_val = torch.clamp(quant_val, 0, 2**bits - 1).to(dtype)

    # De-quantize
    dequantized_tensor = min_val + scale * quant_val.to(tensor.dtype)

    quant_info = {
        'min_val': min_val,
        'scale': scale,
        'quant_val': quant_val
    }
    return quant_info, dequantized_tensor


def quantize_model(model_state_dict, quant_bit, skip_keys=None, axis=0):
    """
    Quantizes the weights of a model.

    Args:
        model_state_dict (dict): The state dictionary of the model.
        quant_bit (int): Number of bits for weight quantization.
        skip_keys (list, optional): A list of substrings. If a key in the
                                               state_dict contains any of these substrings,
                                               it will not be quantized. Defaults to None.

    Returns:
        tuple:
            - quantize_model_info (dict): A dictionary where keys are parameter names
                                             and values are the 'quant_info' dicts from quantize_tensor.
            - dequantized_state_dict (dict): A new state dictionary with de-quantized weights,
                                             ready to be loaded into a model for evaluation.
    """
    if skip_keys is None:
        skip_keys = []

    quantize_model_info = {}
    dequantized_state_dict = deepcopy(model_state_dict) # Start with original weights

    for k, v_tensor in model_state_dict.items():
        skip = False
        for skip_str in skip_keys:
            if skip_str in k:
                skip = True
                break
        
        if skip:
            # For skipped keys, quant_info can indicate no quantization and store original
            quantize_model_info[k] = {
                'min_val': torch.tensor(0.0, device=v_tensor.device),
                'scale': torch.tensor(1.0, device=v_tensor.device),
                'quant_val': v_tensor.clone(), # Store original tensor
                'skipped': True
            }
            # dequantized_state_dict[k] is already the original tensor due to deepcopy
            continue

        if v_tensor.is_floating_point(): # Only quantize floating point tensors
            # Assuming per-tensor quantization for weights as in the original logic for 'else' branch
            quant_info, dequant_v = quantize_tensor(v_tensor, quant_bit, axis=axis)
            quantize_model_info[k] = quant_info
            dequantized_state_dict[k] = dequant_v
        else: # For non-floating point tensors (e.g., batchnorm running_mean/var if not skipped)
             quantize_model_info[k] = {
                'min_val': torch.tensor(0.0, device=v_tensor.device),
                'scale': torch.tensor(1.0, device=v_tensor.device),
                'quant_val': v_tensor.clone(),
                'skipped': True # Effectively skipped as not FP
            }
            # dequantized_state_dict[k] is already the original

    return quantize_model_info, dequantized_state_dict