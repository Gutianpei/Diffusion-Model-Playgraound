import os,sys
os.makedirs("checkpoint", exist_ok=True)
os.makedirs("precomputed", exist_ok=True)
os.makedirs("pretrained", exist_ok=True)
os.makedirs("runs", exist_ok=True)
os.makedirs("runs/classifier-eval", exist_ok=True)
sys.path.append("./")

from datasets.data_utils import get_dataset, get_dataloader
from ourddpm import OurDDPM
from utils.image_utils import fuse, normalize
from main import dict2namespace
import argparse
import yaml
import warnings
warnings.filterwarnings(action='ignore')
from PIL import Image
import cv2


import torch
import pickle
import torchvision.utils as tvu
import torch.nn.functional as F

# # parser = argparse.ArgumentParser(description='.')
# # parser.add_argument('img_index', metavar='N', type=int, nargs=1,
# #                     help='index of the image')
# # args = parser.parse_args()

def get_scheduler(s):
    scheduler = set()
    periods = s.split(",")
    for p in periods:
        l = int(p.split("-")[0])
        r = int(p.split("-")[1])
        for i in range(l, r):
            scheduler.add(i)
    return scheduler


model_path = os.path.join("pretrained/celeba_hq.ckpt")

exp_dir = f"runs/guided"
os.makedirs(exp_dir, exist_ok=True)

# img_index = args.img_index[0]
n_step =  999#@param {type: "integer"}
sampling = "ddpm" #@param ["ddpm", "ddim"]
fixed_xt = True #@param {type: "boolean"}
add_var = True #@param {type: "boolean"}
add_var_on = "0-999" #@param {type: "string"}
vis_gen =  True #@param {type: "boolean"}
guidance_on  = "800-999"

var_scheduler = get_scheduler(add_var_on)
guidance_scheduler = get_scheduler(guidance_on)

args_dic = {
    'config': 'celeba.yml', 
    'n_step': int(n_step), 
    'sample_type': sampling, 
    'eta': 0.0,
    'bs_train': 1, 
    'model_path': model_path, 
    'hybrid_noise': 0, 
    'align_face': 0,
    'image_folder': exp_dir,
    'add_var': bool(add_var),
    'add_var_on': add_var_on,
    'guidance_scheduler': guidance_scheduler
    }
args = dict2namespace(args_dic)

with open(os.path.join('configs', args.config), 'r') as f:
    config_dic = yaml.safe_load(f)
config = dict2namespace(config_dic)

device = torch.device("cuda")

config.device = device
runner = OurDDPM(args, config, device=device)
runner.load_classifier("checkpoint/attr_classifier_4_attrs_40.pt", feature_num=4)

data_root = "/home/summertony717/data/celeba_hq"
train_dataset, val_dataset, test_dataset = get_dataset("CelebA_HQ", data_root, runner.config)#, label = "../DMP_data/list_attr_celeba.csv.zip")
loader_dic = get_dataloader(train_dataset, test_dataset, bs_train=runner.args.bs_train,
                                    num_workers=runner.config.data.num_workers, multi_proc=False,
                                    rank=0, world_size=1)
test_loader = loader_dic["test"]

res = []
for t in range(1, 101):
# for t in range(1,2):
    acc, _ = runner.eval_classifier(test_loader, t0=t)
    res.append(acc)


import pickle
with open('runs/classifier-eval/results.pickle', 'wb') as f:
    pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)


