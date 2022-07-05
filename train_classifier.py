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

warnings.filterwarnings(action='ignore')

device = 'cuda'

args_dic = {
    'config': 'celeba.yml',
    'bs_train': 4,
    'device': device
    }
args = dict2namespace(args_dic)

with open(os.path.join('configs', args.config), 'r') as f:
    config_dic = yaml.safe_load(f)
config = dict2namespace(config_dic)

runner = OurDDPM(args, config, device=device)


print("Start training")
runner.train_classifier()
torch.save(runner.model.state_dict(), "gender_classifier.pt")
