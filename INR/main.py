import json
import os
import torch
import time
import tqdm
import scipy.io
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import OrderedDict
from dahuffman import HuffmanCodec

from models import *
from utils import *
from opt import parse_args, setup_environment, setup_logging

class Trainer:
    def __init__(self, model, lr=1e-4):

        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_func = torch.nn.MSELoss()
        
        # 存储训练过程中的最佳指标
        self.best_vals = {'psnr': 0.0, 'loss': 1e8}
        
        # 记录训练日志
        self.logs = {'psnr': [], 'loss': []}
        
        # 存储最佳模型的参数（以达到的最高PSNR为准）
        self.best_model = OrderedDict((k, v.detach().clone()) for k, v in self.model.state_dict().items())

        # 打印参数信息
        self._print_model_info()

    def _print_model_info(self):
        """打印参数分布信息"""
        net_params_count = 0
        other_params_count = 0

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name.startswith('net.'):
                    net_params_count += param.numel()
                else:
                    other_params_count += param.numel()
        
        total_params_count = net_params_count + other_params_count
        net_params_percentage = (net_params_count / total_params_count) * 100 if total_params_count > 0 else 0
        other_params_percentage = (other_params_count / total_params_count) * 100 if total_params_count > 0 else 0

        print(f"总参数量: {total_params_count/1e6:.4f}M")
        print(f"  - Net 参数量: {net_params_count/1e6:.4f}M ({net_params_percentage:.2f}%)")
        print(f"  - Embedding 参数量: {other_params_count/1e6:.4f}M ({other_params_percentage:.2f}%)")

    def train(self, coordinates, features, num_iters):
        with tqdm.trange(num_iters, ncols=100) as t:
            for _ in t:
                # 前向传播和计算损失
                self.optimizer.zero_grad()
                predicted = self.model(coordinates)
                loss = self.loss_func(predicted, features)
                
                # 反向传播和参数更新
                loss.backward()
                self.optimizer.step()

                # 计算PSNR
                psnr = compute_psnr(predicted, features)

                # 打印结果并更新日志
                log_dict = {
                    'loss': loss.item(),
                    'psnr': psnr,
                    'best_psnr': self.best_vals['psnr']
                }
                t.set_postfix(**log_dict)
                
                # 记录训练日志
                for key in ['loss', 'psnr']:
                    self.logs[key].append(log_dict[key])

                # 更新最佳值
                if loss.item() < self.best_vals['loss']:
                    self.best_vals['loss'] = loss.item()
                
                if psnr > self.best_vals['psnr']:
                    self.best_vals['psnr'] = psnr
                    # 保存最佳模型
                    for k, v in self.model.state_dict().items():
                        self.best_model[k].copy_(v)

def create_model(args, device):
    if args.model_type == 'SPINR':
        return SPINR(table_length=args.table_length, table_dim=args.table_dim, 
                     out_features=args.out_features, hidden_features=args.hidden_features, hidden_layers=args.hidden_layers, 
                     first_omega=args.first_omega, hidden_omega=args.hidden_omega ).to(device)

    elif args.model_type == 'Siren':
        return Siren(in_features=args.in_features, out_features=args.out_features, hidden_features=args.hidden_features, hidden_layers=args.hidden_layers, 
                     first_omega=args.first_omega,hidden_omega=args.hidden_omega).to(device)
    
    elif args.model_type == 'FINER':
        return FINER(in_features=args.in_features, out_features=args.out_features, hidden_layers=args.hidden_layers, hidden_features=args.hidden_features,
                     first_omega=args.first_omega, hidden_omega=args.hidden_omega).to(device)
    
    elif args.model_type == 'PEMLP':
        return PEMLP(in_features=args.in_features, out_features=args.out_features, hidden_layers=args.hidden_layers, hidden_features=args.hidden_features,
                     N_freqs=args.N_freqs).to(device)
    
    elif args.model_type == 'Gauss':
        return Gauss(in_features=args.in_features, out_features=args.out_features, hidden_layers=args.hidden_layers, hidden_features=args.hidden_features,
                     scale=args.scale)
    

def get_rec_img(model, coordinates, img_shape):
    """通过模型获取重建图像"""
    with torch.no_grad():
        rec_features = model(coordinates)   # rec_features = (N, bands), N = height * width
        # Reshape: [N,bands]->[height, width, bands], then permute: [height, width, bands->[bands, height, width]
        rec_img = rec_features.reshape(img_shape[1], img_shape[2], img_shape[0]).permute(2, 0, 1)
    return rec_features, rec_img

def calculate_metrics(rec_features, features, rec_img, img):
    """计算评估指标"""
    psnr = compute_psnr(rec_features, features)
    ssim = compute_ssim(rec_img, img)
    sam = compute_sam(rec_img, img)
    return {'psnr': psnr, 'ssim': ssim, 'sam': sam}

def print_initial_info(img, args, img_name):
    """打印初始化信息"""
    print("-" * 80)

    print(f"数据集: {img_name}")
    print(f"维度: [波段: {img.shape[0]}, 高度: {img.shape[1]}, 宽度: {img.shape[2]}]")
    print(f"模型类型: {args.model_type}")
    print(f"网络结构: [{args.hidden_layers} , {args.hidden_features}]")

    if args.use_quant:
        print(f"启用量化, 量化位数: {args.quant_bits} bit")
    
    if args.use_encode:
        print(f"启用熵编码")

    print("-" * 80)

def train_model(model, coordinates, features, args):
    """训练模型"""
    trainer = Trainer(model, lr=args.learning_rate)
    print("\n开始训练模型...")
    start_time = time.time()
    trainer.train(coordinates, features, num_iters=args.num_iters)
    encode_time = time.time() - start_time
    print(f"训练完成，用时: {encode_time:.4f}秒")
    return trainer, encode_time

def evaluate_fp_model(best_model_state_dict, args, img, coordinates, features,
                      logdir, exp_name, original_file_size_bytes, device, dtype, encode_time):
    """评估全精度模型"""
    model = create_model(args, device)
    model.load_state_dict(best_model_state_dict)
    model = model.to(device, dtype)
    model.eval()

    model_path = os.path.join(logdir, f"{exp_name}_fp.pth")
    torch.save(model.state_dict(), model_path)
    model_size_bytes = os.path.getsize(model_path)

    # --- 测量10次解码时间的平均值 ---
    num_decode_runs = 10
    total_decode_time = 0
    
    # 1. 热身 (Warm-up)
    _, _ = get_rec_img(model, coordinates, img.shape)
    
    # 2. 循环计时
    for _ in range(num_decode_runs):
        start_time = time.time()
        rec_features, rec_img = get_rec_img(model, coordinates, img.shape)
        torch.cuda.synchronize() # 确保GPU操作完成
        end_time = time.time()
        total_decode_time += (end_time - start_time)
        
    # 3. 计算平均值
    decode_time = total_decode_time / num_decode_runs

    metrics = calculate_metrics(rec_features, features, rec_img, img)
    pth_bpppb = (model_size_bytes * 8) / img.numel()
    total_bits = sum(p.numel() for p in model.parameters()) * 32
    final_bpppb = total_bits / img.numel()
    compression_ratio = original_file_size_bytes / model_size_bytes

    eval_results_fp = {
        'pth_bpppb': pth_bpppb,
        'final_bpppb':final_bpppb,
        **metrics,
        'compression_ratio': compression_ratio,
        'model_size_kb': model_size_bytes / 1024,
        'encode_time': encode_time,
        'decode_time': decode_time
    }
    return eval_results_fp, rec_img

def evaluate_quantized_model(best_model_state_dict, args, img, coordinates, features,
                             logdir, exp_name, original_file_size_bytes, device, dtype, encode_time):
    """评估量化模型"""
    if not args.use_quant: return None, None

    model = create_model(args, device)
    model.load_state_dict(best_model_state_dict)
    model = model.to(device, dtype)
    model.eval()

    quantize_model_info, dequantized_state_dict = quantize_model(model.state_dict(), args.quant_bits, args.skip_keys, args.axis)
    model.load_state_dict(dequantized_state_dict)
    quant_method = f"Q{args.quant_bits}"

    quantized_model_path = os.path.join(logdir, f"{exp_name}_{quant_method}.pth")
    torch.save(quantize_model_info, quantized_model_path)
    model_size_bytes = os.path.getsize(quantized_model_path)
    
    # --- 测量10次解码时间的平均值 ---
    num_decode_runs = 10
    total_decode_time = 0
    
    # 1. 热身 (Warm-up)
    _, _ = get_rec_img(model, coordinates, img.shape)
    
    # 2. 循环计时
    for _ in range(num_decode_runs):
        start_time = time.time()
        rec_features_q, rec_img_q = get_rec_img(model, coordinates, img.shape)
        torch.cuda.synchronize() # 确保GPU操作完成
        end_time = time.time()
        total_decode_time += (end_time - start_time)
        
    # 3. 计算平均值
    decode_time = total_decode_time / num_decode_runs

    total_bits = 0
    for layer_info in quantize_model_info.values():
        if layer_info.get('skipped', False): total_bits += layer_info['quant_val'].numel() * 32
        else:
            total_bits += layer_info['quant_val'].numel() * args.quant_bits
            if layer_info.get('min_val') is not None: total_bits += layer_info['min_val'].numel() * 32
            if layer_info.get('scale') is not None: total_bits += layer_info['scale'].numel() * 32
    final_bpppb = total_bits / img.numel()

    if args.use_encode:
        quant_method += "_encoded"
        quant_v_list = []
        for k, layer_info in quantize_model_info.items():
            if not layer_info.get('skipped', False) and not layer_info['quant_val'].dtype.is_floating_point:
                quant_v_list.extend(layer_info['quant_val'].flatten().tolist())

        # get the element name and its frequency
        unique, counts = np.unique(quant_v_list, return_counts=True)
        num_freq = dict(zip(unique, counts))

        # generating HuffmanCoding table
        codec = HuffmanCodec.from_data(quant_v_list)
        sym_bit_dict = {}
        for k, v in codec.get_code_table().items():
            sym_bit_dict[k] = v[0]

        total_bits = 0
        for num, freq in num_freq.items():
            total_bits += freq * sym_bit_dict[num]

        # 加上所有层的开销 (min/scale for quantized, full bits for skipped)
        for layer_info in quantize_model_info.values():
            if layer_info.get('skipped', False): total_bits += layer_info['quant_val'].numel() * 32
            else:
                if layer_info.get('min_val') is not None: total_bits += layer_info['min_val'].numel() * 32
                if layer_info.get('scale') is not None: total_bits += layer_info['scale'].numel() * 32
        final_bpppb = total_bits / img.numel()

    metrics = calculate_metrics(rec_features_q, features, rec_img_q, img)
    pth_bpppb = (model_size_bytes * 8) / img.numel()
    compression_ratio = original_file_size_bytes / model_size_bytes

    eval_results_q = {
        'quant_method': quant_method,
        'pth_bpppb': pth_bpppb,
        'final_bpppb':final_bpppb,
        **metrics,
        'compression_ratio': compression_ratio,
        'model_size_kb': model_size_bytes / 1024,
        'encode_time': encode_time,
        'decode_time': decode_time
    }
    return eval_results_q, rec_img_q

def print_results_summary(eval_results):
    """打印评估结果摘要"""
    print("-" * 80)
    print("结果摘要:")
    print("-" * 80)

    fp_results = eval_results.get('fp')
    print("FP metrics:")
    print(f"  pth_bpppb: {fp_results['pth_bpppb']:.4f}")
    print(f"  final_bpppb: {fp_results['final_bpppb']:.4f}")
    print(f"  PSNR: {fp_results['psnr']:.4f} dB")
    print(f"  SSIM: {fp_results['ssim']:.4f}")
    print(f"  SAM: {fp_results['sam']:.4f} rad")
    print(f"  压缩比: {fp_results['compression_ratio']:.2f} x")
    print(f"  模型大小: {fp_results['model_size_kb']:.2f} KB")
    print(f"  编码时间: {fp_results['encode_time']:.4f} s")
    print(f"  解码时间: {fp_results['decode_time']:.4f} s")

    print("-" * 80)

    quant_results = eval_results.get('quantized')
    if quant_results:
        print(f"Quantized metrics ({quant_results['quant_method']}):")
        print(f"  pth_bpppb: {quant_results['pth_bpppb']:.4f}")
        print(f"  final_bpppb: {quant_results['final_bpppb']:.4f}")
        print(f"  PSNR: {quant_results['psnr']:.4f} dB")
        print(f"  SSIM: {quant_results['ssim']:.4f}")
        print(f"  SAM: {quant_results['sam']:.4f} rad")
        print(f"  压缩比: {quant_results['compression_ratio']:.2f} x")
        print(f"  模型大小: {quant_results['model_size_kb']:.2f} KB")
        print(f"  编码时间: {quant_results['encode_time']:.4f} s")
        print(f"  解码时间: {quant_results['decode_time']:.4f} s")
    else:
        print("量化模型: N/A")

    print("-" * 80)

def save_all_results(training_logs, rec_img_fp, rec_img_q,
                     args, logdir, exp_name, eval_results):
    """保存所有结果，包括图像、指标和日志"""

    # 1. 保存重建图像
    if rec_img_fp is not None:
        img_np_fp = rec_img_fp.cpu().numpy().transpose(1, 2, 0)
        if args.save_rec_img:
            mat_path_fp = os.path.join(logdir, f"{exp_name}_rec_fp.mat")
            scipy.io.savemat(mat_path_fp, {'data': img_np_fp})
            print(f"全精度模型解压缩得到的.mat文件保存到 {mat_path_fp}")

    if rec_img_q is not None:
        img_np_q = rec_img_q.cpu().numpy().transpose(1, 2, 0)
        quant_method = eval_results['quantized']['quant_method']
        if args.save_rec_img:
            mat_path_q = os.path.join(logdir, f"{exp_name}_rec_{quant_method}.mat")
            scipy.io.savemat(mat_path_q, {'data': img_np_q})
            print(f"量化模型解压缩得到的.mat文件保存到 {mat_path_q}")

    # 2. 保存指标 (JSON)
    results_to_save = {
        'results': eval_results,
        'args': vars(args)
    }
    with open(os.path.join(logdir, f"{exp_name}_results_Q{args.quant_bits}.json"), 'w') as f:
        json.dump(results_to_save, f, indent=4, default=str)

    # 3. 保存训练曲线和日志
    if training_logs:
        plt.figure(figsize=(10, 6))
        plt.plot(training_logs['psnr'], color='#2E86C1', linewidth=2)
        
        plt.title('PSNR Training Curve', fontsize=14, pad=15)
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('PSNR (dB)', fontsize=12)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tick_params(direction='out', length=6, width=1)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(1.5)
        plt.gca().set_facecolor('#F8F9F9')
        plt.tight_layout()
        plt.savefig(os.path.join(logdir, f"{exp_name}_training_curves.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()

        with open(os.path.join(logdir, f"{exp_name}_training_logs.json"), 'w') as f:
            json.dump(training_logs, f, indent=4, default=str)

    print(f"实验结果已保存到 {logdir}")
    print("-" * 80)


def main():
    """主函数，实现高光谱图像压缩的全流程"""
    # 1. 初始化设置
    args = parse_args()
    device, dtype = setup_environment(args)

    # 2. 数据加载与准备
    img = load_hsi_data(args.img_path, device, dtype) # img is (bands, height, width)
    args.table_length = img.shape[1] * img.shape[2] # table_length = height * width
    args.out_features = img.shape[0] # out_features = bands
    img_name = os.path.basename(args.img_path).split('.')[0]
    logdir, exp_name = setup_logging(args, img_name)
    coordinates, features = to_coordinates_and_features(img) # coordinates (H*W, 2), features (H*W, C)
    original_file_size_bytes = os.path.getsize(args.img_path)

    # 3. 创建初始模型
    model = create_model(args, device)

    # 4. 打印初始信息
    print_initial_info(img, args, img_name)

    # 5. 训练模型 或 加载预训练模型
    model_loaded_from_file = False
    best_model_state = None
    encode_time = 0.0
    training_logs = {}
    trainer = None


    default_fp_model_path = os.path.join(logdir, f"{exp_name}_fp.pth")
    if os.path.exists(default_fp_model_path) and args.load_cpt:
        print(f"正在从 {default_fp_model_path} 加载预训练模型...")
        try:
            best_model_state = torch.load(default_fp_model_path, map_location=device, weights_only=False)
            model_loaded_from_file = True
            print(f"从 {default_fp_model_path} 模型加载成功.")

            # 尝试加载上次的编码时间和训练日志
            results_json_path = os.path.join(os.path.dirname(default_fp_model_path), f"{exp_name}_results_Q8.json")
            training_logs_json_path = os.path.join(os.path.dirname(default_fp_model_path), f"{exp_name}_training_logs.json")
            if os.path.exists(results_json_path):
                with open(results_json_path, 'r') as f_results:
                    saved_results_data = json.load(f_results)
                # 尝试获取编码时间
                if 'results' in saved_results_data and 'fp' in saved_results_data['results'] and 'encode_time' in saved_results_data['results']['fp']:
                    encode_time = saved_results_data['results']['fp']['encode_time']
                else:
                    print(f"在 {results_json_path} 中未找到上次的编码时间, 将使用0.")
                    encode_time = 0.0
            else:
                print(f"未找到结果文件 {results_json_path}, 编码时间将为0.")
                encode_time = 0.0

            if os.path.exists(training_logs_json_path):
                with open(training_logs_json_path, 'r') as f_logs:
                    training_logs = json.load(f_logs)
                print(f"从 {training_logs_json_path} 加载上次训练日志.")
            else:
                print(f"未找到训练日志文件 {training_logs_json_path}, 训练日志将为空.")
                training_logs = {}

        except Exception as e:
            print(f"从 {default_fp_model_path} 加载模型失败: {e}. 将重新训练模型.")
            model_loaded_from_file = False
            encode_time = 0.0
            training_logs = {}
            model = create_model(args, device)
    
    if not model_loaded_from_file:
        trainer, encode_time = train_model(model, coordinates, features, args)
        best_model_state = trainer.best_model 
        training_logs = trainer.logs
    
    # 6. 评估模型
    eval_results_fp, rec_img_fp = evaluate_fp_model(
        best_model_state, args, img, coordinates, features,
        logdir, exp_name, original_file_size_bytes, device, dtype, encode_time
    )

    eval_results_q, rec_img_q = None, None
    if args.use_quant:
        eval_results_q, rec_img_q = evaluate_quantized_model(
            best_model_state, args, img, coordinates, features,
            logdir, exp_name, original_file_size_bytes, device, dtype, encode_time
        )

    all_eval_results = {
        'fp': eval_results_fp,
        'quantized': eval_results_q
    }

    # 7. 打印结果摘要
    print_results_summary(all_eval_results)

    # 8. 保存结果
    save_all_results(training_logs, rec_img_fp, rec_img_q,
                     args, logdir, exp_name, all_eval_results)

if __name__ == "__main__":
    main()