"""Command-line argument parsing and environment setup
Provides command-line argument parsing, environment setup, and logging configuration
"""

import argparse
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='HSI compression based on INRs argument settings')
    
    ## train
    parser.add_argument("--num_iters", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--load_cpt", action="store_true", default=False)

    ## model
    parser.add_argument('--in_features', type=int, default=2)
    parser.add_argument('--out_features', type=int, default=3)  
    parser.add_argument('--hidden_layers', type=int, default=3) 
    parser.add_argument('--hidden_features', type=int, default=256)
    # 
    parser.add_argument('--model_type', type=str, default='SPINR', required=['SPINR','Siren','Finer','PEMLP','Gauss'])
    parser.add_argument('--first_omega', type=float, default=30)
    parser.add_argument('--hidden_omega', type=float, default=30)
    parser.add_argument('--scale', type=float, default=30)
    parser.add_argument('--N_freqs', type=int, default=10)
    # 
    parser.add_argument("--table_dim", type=int, default=2)

    ## environment
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=0)

    ## data
    parser.add_argument("--img_path", type=str, required=True) 
    parser.add_argument("--save_rec_img", action="store_true", default=False) 

    ## compression
    parser.add_argument("--use_quant", action="store_true", default=False)
    parser.add_argument("--skip_keys", type=str, nargs='*', default=[])
    parser.add_argument("--axis", type=int, default=0)
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument("--use_encode", action="store_true", default=False)
    
    return parser.parse_args()

def setup_environment(args):
    # Set the specified GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    # Set default device and default data type
    dtype = torch.float32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    return device, dtype

def setup_logging(args, img_name):
    # Set experiment name
    exp_name = f"{img_name}_{args.model_type}_{args.hidden_layers}×{args.hidden_features}"
    if args.model_type == 'SPINR':
        exp_name += f"_Tdim{args.table_dim}"
    if args.model_type == 'Gauss':
        exp_name += f"_scale{args.scale}"
    if args.model_type == 'PEMLP':
        exp_name += f"_Nf{args.N_freqs}"
    # if args.use_quant:
    #     exp_name += f"_Q{args.quant_bits}"
    exp_name += f"_it{args.num_iters}"
    # Create log directory
    logdir = os.path.join('results/single_exp', exp_name)
    os.makedirs(logdir, exist_ok=True)
    
    return logdir, exp_name