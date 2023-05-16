import time
import random
import torch
torch.cuda.current_device()
import logging
import argparse
import os
import numpy as np
import warnings

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

localtime = time.localtime(time.time())
str_time = f'{str(localtime.tm_mon)}-{str(localtime.tm_mday)}-{str(localtime.tm_hour)}-{str(localtime.tm_min)}'

from model.model import MMC
from src.train_food101 import train_food101
from src.config import Config
from src.functions import dict_to_str

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run(args):
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.fig_save_dir):
        os.makedirs(args.fig_save_dir)

    args.name_seed = args.name + '_' + str(args.seed)
    args.model_save_path = os.path.join(args.model_save_dir, f'{args.dataset}-{args.name_seed}-{str_time}.pth')
    args.best_model_save_path = os.path.join(args.model_save_dir,  f'{args.dataset}-{args.name_seed}-best.pth')
    
    setup_seed(args.seed)
    if args.dataset in ['Food101', 'N24News']:
        results = train_food101(args)
    else:
        results = train_medical(args)
    return results

def set_log(args):
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    log_file_path = os.path.join(args.logs_dir, f'{args.dataset}-{args.name}-{str_time}.log')
    # set logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # logger.setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    for ph in logger.handlers:
        logger.removeHandler(ph)
    # add FileHandler to log file
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    # add StreamHandler to terminal outputs
    formatter_stream = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)
    return logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='MMC',
                        help='project name')
    parser.add_argument('--dataset', type=str, default='Food101',
                        help='support N24News/Food101')
    parser.add_argument('--text_type', type=str, default='headline',
                        help='support headline/caption/abstract')
    parser.add_argument('--mmc', type=str, default='UniSMMC',
                        help='support UniSMMC/UnSupMMC/SupMMC')
    parser.add_argument('--mmc_tao', type=float, default=0.07,
                        help='use supervised contrastive loss or not')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--lr_mm', type=float, default=1e-3,
                        help='--lr_mm')
    parser.add_argument('--min_epoch', type=int, default=1,
                        help='min_epoch')    
    parser.add_argument('--valid_step', type=int, default=50,
                        help='valid_step')              
    parser.add_argument('--max_length', type=int, default=512,
                        help='max_length')
    parser.add_argument('--text_encoder', type=str, default='bert_base',
                        help='bert_base/roberta_base/bert_large')
    parser.add_argument('--image_encoder', type=str, default='vit_base',
                        help='vit_base/vit_large')
    parser.add_argument('--text_out', type=int, default=768,
                        help='text_out')
    parser.add_argument('--img_out', type=int, default=768,
                        help='img_out')                                        
    parser.add_argument('--lr_mm_cls', type=float, default=1e-3,
                        help='--lr_mm_cls')
    parser.add_argument('--mm_dropout', type=float, default=0.0,
                        help='--mm_dropout')
    parser.add_argument('--lr_text_tfm', type=float, default=2e-5,
                        help='--lr_text_tfm')
    parser.add_argument('--lr_img_tfm', type=float, default=5e-5,
                        help='--lr_img_tfm')
    parser.add_argument('--lr_img_cls', type=float, default=1e-4,
                        help='--lr_img_cls')
    parser.add_argument('--lr_text_cls', type=float, default=5e-5,
                        help='--lr_text_cls')
    parser.add_argument('--text_dropout', type=float, default=0.0,
                        help='--text_dropout')
    parser.add_argument('--img_dropout', type=float, default=0.1,
                        help='--img_dropout')
    parser.add_argument('--nplot', type=str, default='',
                        help='MTAV')
    parser.add_argument('--data_dir', type=str, default='Path/To/Dataset_Home_Directory/',
                        help='support wmsa') 
    parser.add_argument('--test_only', type=bool, default=False,
                        help='train+test or test only')
    parser.add_argument('--pretrained_dir', type=str, default='Path/To/Pretrained',
                        help='path to pretrained models from Hugging Face.')
    parser.add_argument('--model_save_dir', type=str, default='Path/To/results/models',
                        help='path to save model parameters.')
    parser.add_argument('--res_save_dir', type=str, default='Path/To/results/results',
                        help='path to save training results.')
    parser.add_argument('--fig_save_dir', type=str, default='Path/To/results/imgs',
                        help='path to save figures.')
    parser.add_argument('--logs_dir', type=str, default='Path/To/results/logs',
                        help='path to log results.')  # NO
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--seeds', nargs='+', type=int,
                        help='set seeds for multiple runs!')
    return parser.parse_args()

if __name__ == '__main__':

    start = time.time()
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore")

    args = parse_args()
    logger = set_log(args)
    config = Config(args)
    args = config.get_config()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")

    args.device = device
    args.data_dir = os.path.join(args.data_dir, args.dataset)

    if args.local_rank in [-1, 0]:
        logger.info("Pytorch version: " + torch.__version__)
        logger.info("CUDA version: " + torch.version.cuda)
        logger.info(f"CUDA device: + {torch.cuda.current_device()}")
        logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
        logger.info("GPU name: " + torch.cuda.get_device_name())
        logger.info("Current Hyper-Parameters:")
        logger.info(args)

    final_results = {}
    final_std_results = {}
    temp_results = {}

    # final_std_results
    for seed in args.seeds:
        args.seed = seed
        temp_results = run(args)
        if len(final_results.keys()):
            for key in temp_results.keys():
                final_results[key] += temp_results[key]
                final_std_results[key].append(temp_results[key])
        else:
            final_results = temp_results
            final_std_results = {key: [] for key in temp_results.keys()}
            for key in temp_results.keys():
                final_std_results[key].append(temp_results[key])

    if args.local_rank in [-1, 0]:
        logger.info(f"Run {len(args.seeds)} times！Final test results:")

        for key in final_results.keys():
            print(key, ": ", final_std_results[key])
            final_std_results[key] = np.std(final_std_results[key])
            final_results[key] = final_results[key] / len(args.seeds)

        logger.info(f"{args.dataset}-{args.name}")
        logger.info("Average: " + dict_to_str(final_results))
        logger.info("Standard deviation: " + dict_to_str(final_std_results))

    end = time.time()

    logger.info(f"Run {end - start} seconds in total！")
