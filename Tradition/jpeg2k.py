import os
import time
import numpy as np
import torch
import subprocess
import tempfile
import shutil
import scipy.io as sio
import json
from pathlib import Path
from utils import load_hsi_data, compute_psnr, compute_ssim, compute_sam

def check_opj_dependency():
    if shutil.which("opj_compress") is None or shutil.which("opj_decompress") is None:
        raise FileNotFoundError("OpenJPEG命令 (opj_compress/opj_decompress) 未找到. 请确保已安装OpenJPEG并将其添加到系统PATH。")

def convert_hsi_to_raw(hsi_data, raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    bands, height, width = hsi_data.shape
    raw_files = []

    data_min, data_max = hsi_data.min(), hsi_data.max()
    if data_max == data_min: data_max += 1e-6

    for band in range(bands):
        band_data = hsi_data[band]
        normalized = ((band_data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        raw_path = os.path.join(raw_dir, f"band_{band:03d}.raw")
        normalized.tofile(raw_path)
        raw_files.append(raw_path)
        
    metadata = {'shape': hsi_data.shape, 'data_min': data_min, 'data_max': data_max, 'nbytes': hsi_data.nbytes}
    np.save(os.path.join(raw_dir, "metadata.npy"), metadata)
    return raw_files

def compress_with_jpeg2k(raw_files, j2k_dir, cr, metadata):
    os.makedirs(j2k_dir, exist_ok=True)
    j2k_files = []
    _, height, width = metadata['shape']
    format_str = f"{width},{height},1,8,u"
    
    for raw_file in raw_files:
        j2k_path = os.path.join(j2k_dir, f"{Path(raw_file).stem}.j2k")
        cmd = ["opj_compress", "-i", raw_file, "-o", j2k_path, "-r", str(cr), "-F", format_str]
        subprocess.run(cmd, check=True, capture_output=True)
        j2k_files.append(j2k_path)
    return j2k_files

def decompress_jpeg2k_to_hsi(j2k_files, metadata_file):
    metadata = np.load(metadata_file, allow_pickle=True).item()
    bands, height, width = metadata['shape']
    data_min, data_max = metadata['data_min'], metadata['data_max']
    reconstructed = np.zeros((bands, height, width), dtype=np.float32)

    with tempfile.TemporaryDirectory() as dec_dir:
        for i, j2k_file in enumerate(sorted(j2k_files)):
            dec_raw_path = os.path.join(dec_dir, f"dec_band_{i:03d}.raw")
            cmd = ["opj_decompress", "-i", j2k_file, "-o", dec_raw_path]
            subprocess.run(cmd, check=True, capture_output=True)
            band_data = np.fromfile(dec_raw_path, dtype=np.uint8).reshape(height, width)
            band_float = (band_data / 255.0) * (data_max - data_min) + data_min
            reconstructed[i] = band_float
    return reconstructed

def jpeg2k_compress_hsi(hsi_path, output_dir, compression_ratios, device, save_mat):
    check_opj_dependency()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.makedirs(output_dir, exist_ok=True)
        hsi_tensor = load_hsi_data(hsi_path, device=device)
        hsi_data = hsi_tensor.cpu().numpy()
        
        raw_dir = os.path.join(temp_dir, "raw")
        raw_files = convert_hsi_to_raw(hsi_data, raw_dir)
        metadata_file = os.path.join(raw_dir, "metadata.npy")
        metadata = np.load(metadata_file, allow_pickle=True).item()
        
        results = {k: [] for k in ['compression_ratio_target', 'compression_ratio', 'bpp', 'psnr', 'ssim', 'sam', 'model_size_kb', 'encode_time', 'decode_time']}
        
        for cr in compression_ratios:
            print(f"\nJPEG2k - 目标压缩比: {cr}")
            
            start_time = time.time()
            j2k_dir = os.path.join(temp_dir, f"j2k_cr{cr}")
            j2k_files = compress_with_jpeg2k(raw_files, j2k_dir, cr, metadata)
            encode_time = time.time() - start_time
            
            start_time = time.time()
            rec_hsi_data = decompress_jpeg2k_to_hsi(j2k_files, metadata_file)
            decode_time = time.time() - start_time

            compressed_size = sum(os.path.getsize(f) for f in j2k_files)
            original_size_bytes = metadata['nbytes']
            
            rec_tensor = torch.from_numpy(rec_hsi_data).to(hsi_tensor)
            results['compression_ratio_target'].append(cr)
            results['psnr'].append(compute_psnr(rec_tensor, hsi_tensor))
            results['ssim'].append(compute_ssim(rec_tensor, hsi_tensor))
            results['sam'].append(compute_sam(rec_tensor, hsi_tensor))
            results['compression_ratio'].append(original_size_bytes / compressed_size)
            results['bpp'].append((compressed_size * 8) / hsi_data.size)
            results['model_size_kb'].append(compressed_size / 1024.0)
            results['encode_time'].append(encode_time)
            results['decode_time'].append(decode_time)

            if save_mat:
                image_name = Path(hsi_path).stem
                output_mat = os.path.join(output_dir, f"{image_name}_jpeg2k_cr{int(cr)}.mat")
                sio.savemat(output_mat, {'data': rec_hsi_data.transpose(1, 2, 0)})
        
        image_name = Path(hsi_path).stem
        json_path = os.path.join(output_dir, f"{image_name}_jpeg2k_results.json")
        for k, v in results.items(): results[k] = [float(val) for val in v]
        json_data = {'image_name': image_name, 'results': results}
        with open(json_path, 'w') as f: json.dump(json_data, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='使用JPEG2k压缩高光谱图像')
    parser.add_argument('--image_path', type=str, required=True, help='高光谱图像文件路径(.mat)')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--compression_ratios', type=str, default='10,20,50,100', help='逗号分隔的JPEG2k压缩比列表')
    parser.add_argument('--device', type=str, default='cpu', help='评估时使用的设备')
    parser.add_argument('--save_mat', action='store_true', help='是否保存重建后的MAT文件')
    args = parser.parse_args()
    
    ratios = [float(r) for r in args.compression_ratios.split(',')]
    jpeg2k_compress_hsi(args.image_path, args.output_dir, compression_ratios=ratios, device=args.device, save_mat=args.save_mat)