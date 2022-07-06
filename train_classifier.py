import os

from ourddpm import OurDDPM
from main import dict2namespace
import argparse
import yaml
from PIL import Image
import warnings

import torch
import pdb
import cv2
import glob
import pickle
import torch.multiprocessing as mp

warnings.filterwarnings(action='ignore')


def main():
    device = 'cuda'

    args_dic = {
        'config': 'celeba.yml',
        'bs_train': 8,
        'device': device
        }
    args = dict2namespace(args_dic)

    with open(os.path.join('configs', args.config), 'r') as f:
        config_dic = yaml.safe_load(f)
    config = dict2namespace(config_dic)

    runner = OurDDPM(args, config, device=device)

    GPU_CNT = torch.cuda.device_count()
    SAVE_PATH = "checkpoint/attr_classifier_noisy.pt"

    print("Start training")
    if GPU_CNT > 1:
        mp.spawn(runner.train_classifier, args=(True, GPU_CNT, SAVE_PATH), nprocs=GPU_CNT, join=True)
    else:
        runner.train_classifier()

if __name__ == "__main__":
    main()