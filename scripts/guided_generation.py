import os,sys
os.makedirs("checkpoint", exist_ok=True)
os.makedirs("precomputed", exist_ok=True)
os.makedirs("pretrained", exist_ok=True)
os.makedirs("runs", exist_ok=True)
os.makedirs("runs/guided", exist_ok=True)


sys.path.append("./")
from ourddpm import OurDDPM
from main import dict2namespace
import argparse
import yaml
import warnings
warnings.filterwarnings(action='ignore')


import torch
import pickle
import torchvision.utils as tvu

parser = argparse.ArgumentParser(description='.')
parser.add_argument('scale', metavar='N', type=int, nargs=1,
                    help='gradient scale')
args = parser.parse_args()

device = 'cuda'
model_path = os.path.join("pretrained/celeba_hq.ckpt")

exp_dir = f"runs/guided"
os.makedirs(exp_dir, exist_ok=True)

scale = args.scale[0]
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


device = torch.device("cuda")
config.device = device
runner = OurDDPM(args, config, device=device)
runner.load_classifier("checkpoint/attr_classifier_noisy1.pt")
res_list = []

for i, j in enumerate([1, 5, 10, -1, -5, -10]):
    xt = data_list[1]["xt"]
    noise_traj = torch.tensor(data_list[1]["noise_traj"]).cuda()
    img = runner.guided_generate_ddpm(xt, var_scheduler, runner.classifier, 1, classifier_scale=scale*j, noise_traj=noise_traj)

    tvu.save_image((img + 1) * 0.5, os.path.join(exp_dir, f'guided_scale_no_sig_no_log_{str(scale*j).zfill(5)}.png'))
    torch.cuda.empty_cache()
