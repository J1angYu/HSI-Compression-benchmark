import os
import time
import numpy as np
import torch
import tempfile
import shutil
import scipy.io as sio
import json
from pathlib import Path
from PIL import Image
from utils import load_hsi_data, compute_psnr, compute_ssim, compute_sam

def convert_hsi_to_images(hsi_data, img_dir):
    os.makedirs(img_dir, exist_ok=True)
    bands, height, width = hsi_data.shape
    img_files = []
    
    data_min, data_max = hsi_data.min(), hsi_data.max()
    if data_max == data_min: data_max += 1e-6

    for band in range(bands):
        band_data = hsi_data[band]
        normalized = ((band_data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        img_path = os.path.join(img_dir, f"band_{band:03d}.png")
        Image.fromarray(normalized).save(img_path)
        img_files.append(img_path)
    
    metadata = {'shape': hsi_data.shape, 'data_min': data_min, 'data_max': data_max, 'nbytes': hsi_data.nbytes}
    np.save(os.path.join(img_dir, "metadata.npy"), metadata)
    return img_files

def compress_with_jpeg(img_files, jpg_dir, quality):
    os.makedirs(jpg_dir, exist_ok=True)
    jpg_files = []
    for img_file in img_files:
        jpg_path = os.path.join(jpg_dir, f"{Path(img_file).stem}.jpg")
        Image.open(img_file).save(jpg_path, "JPEG", quality=quality, subsampling=0, optimize=True)
        jpg_files.append(jpg_path)
    return jpg_files

def decompress_jpeg_to_hsi(jpg_files, metadata_file):
    metadata = np.load(metadata_file, allow_pickle=True).item()
    bands, height, width = metadata['shape']
    data_min, data_max = metadata['data_min'], metadata['data_max']
    reconstructed = np.zeros((bands, height, width), dtype=np.float32)
    
    for i, jpg_file in enumerate(sorted(jpg_files)):
        band_data = np.array(Image.open(jpg_file))
        band_float = (band_data / 255.0) * (data_max - data_min) + data_min
        reconstructed[i] = band_float
    return reconstructed

def jpeg_compress_hsi(hsi_path, output_dir, quality_levels, device, save_mat):
    with tempfile.TemporaryDirectory() as temp_dir:
        os.makedirs(output_dir, exist_ok=True)
        hsi_tensor = load_hsi_data(hsi_path, device=device)
        hsi_data = hsi_tensor.cpu().numpy()
        
        img_dir = os.path.join(temp_dir, "png")
        img_files = convert_hsi_to_images(hsi_data, img_dir)
        metadata_file = os.path.join(img_dir, "metadata.npy")
        metadata = np.load(metadata_file, allow_pickle=True).item()
        
        results = {k: [] for k in ['quality', 'compression_ratio', 'bpp', 'psnr', 'ssim', 'sam', 'model_size_kb', 'encode_time', 'decode_time']}
        
        for quality in quality_levels:
            print(f"\nJPEG - 质量: {quality}")
            start_time = time.time()
            jpg_dir = os.path.join(temp_dir, f"jpg_q{quality}")
            jpg_files = compress_with_jpeg(img_files, jpg_dir, quality)
            encode_time = time.time() - start_time
            
            start_time = time.time()
            rec_hsi_data = decompress_jpeg_to_hsi(jpg_files, metadata_file)
            decode_time = time.time() - start_time
            
            compressed_size = sum(os.path.getsize(f) for f in jpg_files)
            original_size_bytes = metadata['nbytes']
            
            rec_tensor = torch.from_numpy(rec_hsi_data).to(hsi_tensor)
            results['quality'].append(quality)
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
                output_mat = os.path.join(output_dir, f"{image_name}_jpeg_q{quality}.mat")
                sio.savemat(output_mat, {'data': rec_hsi_data.transpose(1, 2, 0)})

        image_name = Path(hsi_path).stem
        json_path = os.path.join(output_dir, f"{image_name}_jpeg_results.json")
        for k, v in results.items(): results[k] = [float(val) for val in v]
        json_data = {'image_name': image_name, 'results': results}
        with open(json_path, 'w') as f: json.dump(json_data, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='使用JPEG压缩高光谱图像')
    parser.add_argument('--image_path', type=str, required=True, help='高光谱图像文件路径(.mat)')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--quality_levels', type=str, default='10,20,50,75,90', help='逗号分隔的JPEG质量列表')
    parser.add_argument('--device', type=str, default='cpu', help='评估时使用的设备')
    parser.add_argument('--save_mat', action='store_true', help='是否保存重建后的MAT文件')
    args = parser.parse_args()
    
    qualities = [int(q) for q in args.quality_levels.split(',')]
    jpeg_compress_hsi(args.image_path, args.output_dir, quality_levels=qualities, device=args.device, save_mat=args.save_mat)