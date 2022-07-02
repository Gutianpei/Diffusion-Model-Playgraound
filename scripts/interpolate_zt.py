
import os, sys

os.makedirs("checkpoint", exist_ok=True)
os.makedirs("precomputed", exist_ok=True)
os.makedirs("pretrained", exist_ok=True)
os.makedirs("runs", exist_ok=True)
os.makedirs("runs/interpolation", exist_ok=True)
sys.path.append("./")

from ourddpm import OurDDPM
from main import dict2namespace
import argparse
import yaml
from PIL import Image
import warnings
warnings.filterwarnings(action='ignore')

device = 'cuda'

from math import sqrt
import torch
import pdb
import cv2
import glob
# from google.colab.patches import cv2_imshow
import pickle
import matplotlib.pyplot as plt

model_path = os.path.join("pretrained/celeba_hq.ckpt")

exp_dir = f"runs/raw_ddpm"
os.makedirs(exp_dir, exist_ok=True)

n_step =  999#@param {type: "integer"}
sampling = "ddpm" #@param ["ddpm", "ddim"]
fixed_xt = True #@param {type: "boolean"}
add_var = True #@param {type: "boolean"}
add_var_on = "0-999" #@param {type: "string"}
vis_gen =  True #@param {type: "boolean"}


args_dic = {
    'config': 'celeba.yml', 
    'n_step': int(n_step), 
    'sample_type': sampling, 
    'eta': 0.0,
    'bs_test': 1, 
    'model_path': model_path, 
    'hybrid_noise': 0, 
    'align_face': 0,
    'image_folder': exp_dir,
    'add_var': bool(add_var),
    'add_var_on': add_var_on
    }
args = dict2namespace(args_dic)

with open(os.path.join('configs', args.config), 'r') as f:
    config_dic = yaml.safe_load(f)
config = dict2namespace(config_dic)


if bool(add_var):
    var_scheduler = []
    periods = add_var_on.split(",")
    for period in periods:
        start = int(period.split("-")[0])
        end = int(period.split("-")[1])
        for n in range(start,end):
            var_scheduler.append(n)

with open("runs/interpolation/data_1.obj","rb") as f:
    data_list = pickle.load(f)

def interpolate_zt(zt1, zt2, steps, xt, var_scheduler):
    device = torch.device("cuda")
    config.device = device
    runner = OurDDPM(args, config, device=device)
    res_list = []
    for i in range(steps):
        print(f"Processing {i}th sample...")
        a = i / (steps-1)
        b = (steps-1-i) / (steps-1)
        zt = (a*zt1+b*zt2)/sqrt(a**2+b**2)

        traj, noise_traj = runner.generate_ddpm_and_noise_traj(xt, var_scheduler, mode="use", noise_traj=zt)

        res_list.append(traj[-1])
        torch.cuda.empty_cache()
    return res_list

STEPS = 5
results = []

for i in range(0, 30, 2):
# for i in range(0, 2, 2):
    zt1 = torch.tensor(data_list[i]["noise_traj"])
    zt2 = torch.tensor(data_list[i+1]["noise_traj"])
    res = interpolate_zt(zt1, zt2, STEPS, data_list[i]["xt"], var_scheduler)
    results.append(res)

with open("runs/interpolation/data_1_interpolate_zt.obj","wb") as f:
    pickle.dump(results, f)