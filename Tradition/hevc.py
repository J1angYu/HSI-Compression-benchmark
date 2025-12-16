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
from PIL import Image
from utils import load_hsi_data, compute_psnr, compute_ssim, compute_sam

def check_ffmpeg_dependency():
    """检查FFmpeg是否已安装。"""
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("FFmpeg未找到。请确保已安装并将其添加到系统PATH。")

def convert_hsi_to_images(hsi_data, img_dir):
    """将HSI数据转换为PNG图像序列。"""
    os.makedirs(img_dir, exist_ok=True)
    bands, _, _ = hsi_data.shape
    img_files_pattern = os.path.join(img_dir, "band_%03d.png")
    
    data_min, data_max = hsi_data.min(), hsi_data.max()
    if data_max == data_min: data_max += 1e-6

    for band in range(bands):
        band_data = hsi_data[band]
        normalized = ((band_data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        Image.fromarray(normalized).save(img_files_pattern % band)
    
    metadata = {'shape': hsi_data.shape, 'nbytes': hsi_data.nbytes, 'data_min': data_min, 'data_max': data_max}
    np.save(os.path.join(img_dir, "metadata.npy"), metadata)
    return img_files_pattern

def compress_with_hevc(img_pattern, output_video_path, crf):
    """使用FFmpeg和libx265将图像序列压缩成HEVC视频。"""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "30",
        "-i", img_pattern,
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        # --- 使用 H.265 编码器 ---
        "-c:v", "libx265",
        "-crf", str(crf),
        "-preset", "slow",
        "-pix_fmt", "gray8",
        output_video_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg编码失败! 命令: {' '.join(e.cmd)}\nFFmpeg输出:\n{e.stderr}")
        raise

def decompress_hevc_to_hsi(video_path, dec_img_dir, metadata_file):
    """将HEVC视频解码回图像序列，并重建HSI数据。"""
    os.makedirs(dec_img_dir, exist_ok=True)
    metadata = np.load(metadata_file, allow_pickle=True).item()
    bands, height, width = metadata['shape']
    data_min, data_max = metadata['data_min'], metadata['data_max']
    dec_img_pattern = os.path.join(dec_img_dir, "rec_band_%03d.png")

    cmd = ["ffmpeg", "-y", "-i", video_path, "-vframes", str(bands), dec_img_pattern]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg解码失败! 命令: {' '.join(e.cmd)}\nFFmpeg输出:\n{e.stderr}")
        raise

    decoded_files = sorted(Path(dec_img_dir).glob('rec_band_*.png'))
    if len(decoded_files) != bands:
        raise RuntimeError(f"解码错误：预期解码{bands}帧, 实际得到{len(decoded_files)}帧。")
        
    reconstructed = np.zeros((bands, height, width), dtype=np.float32)
    for i, img_path in enumerate(decoded_files):
        img = Image.open(img_path).convert('L')
        band_data = np.array(img)
        cropped_band_data = band_data[0:height, 0:width]
        band_float = (cropped_band_data / 255.0) * (data_max - data_min) + data_min
        reconstructed[i] = band_float
            
    return reconstructed

def hevc_compress_hsi(hsi_path, output_dir, crf_levels, device, save_mat):
    """完整的高光谱图像HEVC压缩与评估流程。"""
    check_ffmpeg_dependency()
    # 使用with语句确保临时目录总是被清理
    with tempfile.TemporaryDirectory() as temp_dir:
        hsi_tensor = load_hsi_data(hsi_path, device=device)
        hsi_data = hsi_tensor.cpu().numpy()
        
        png_dir = os.path.join(temp_dir, "png_sequence")
        img_pattern = convert_hsi_to_images(hsi_data, png_dir)
        metadata_file = os.path.join(png_dir, "metadata.npy")
        metadata = np.load(metadata_file, allow_pickle=True).item()
        
        results = {k: [] for k in ['crf_target', 'compression_ratio', 'bpp', 'psnr', 'ssim', 'sam', 'model_size_kb', 'encode_time', 'decode_time']}
        
        for crf in crf_levels:
            print(f"\nHEVC - 目标 CRF: {crf}")
            video_path = os.path.join(output_dir, f"{Path(hsi_path).stem}_hevc_crf{crf}.mp4")
            
            start_time = time.time()
            compress_with_hevc(img_pattern, video_path, crf)
            encode_time = time.time() - start_time
            
            start_time = time.time()
            dec_dir = os.path.join(temp_dir, f"dec_crf{crf}")
            rec_hsi_data = decompress_hevc_to_hsi(video_path, dec_dir, metadata_file)
            decode_time = time.time() - start_time

            compressed_size = os.path.getsize(video_path)
            original_size_bytes = metadata['nbytes']
            
            rec_tensor = torch.from_numpy(rec_hsi_data).to(hsi_tensor)
            results['crf_target'].append(crf)
            results['psnr'].append(compute_psnr(rec_tensor, hsi_tensor))
            results['ssim'].append(compute_ssim(rec_tensor, hsi_tensor))
            results['sam'].append(compute_sam(rec_tensor, hsi_tensor))
            results['compression_ratio'].append(original_size_bytes / compressed_size)
            results['bpp'].append((compressed_size * 8) / hsi_data.size)
            results['model_size_kb'].append(compressed_size / 1024.0)
            results['encode_time'].append(encode_time)
            results['decode_time'].append(decode_time)
            
            if save_mat:
                output_mat = os.path.join(output_dir, f"{Path(hsi_path).stem}_hevc_crf{crf}.mat")
                sio.savemat(output_mat, {'data': rec_hsi_data.transpose(1, 2, 0)})
        
        image_name = Path(hsi_path).stem
        os.makedirs(output_dir, exist_ok=True) # 确保输出目录存在
        json_path = os.path.join(output_dir, f"{image_name}_hevc_results.json")
        for k, v in results.items(): results[k] = [float(val) for val in v]
        json_data = {'image_name': image_name, 'results': results}
        with open(json_path, 'w') as f: json.dump(json_data, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='使用HEVC (H.265) 压缩高光谱图像')
    parser.add_argument('--image_path', type=str, required=True, help='高光谱图像文件路径(.mat)')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--crf_levels', type=str, default='23,28,32,37', help='逗号分隔的HEVC CRF值列表')
    parser.add_argument('--device', type=str, default='cpu', help='评估时使用的设备')
    parser.add_argument('--save_mat', action='store_true', help='是否保存重建后的MAT文件')
    args = parser.parse_args()
    
    crf_values = [int(q) for q in args.crf_levels.split(',')]
    hevc_compress_hsi(args.image_path, args.output_dir, crf_levels=crf_values, device=args.device, save_mat=args.save_mat)