import os,sys
os.makedirs("checkpoint", exist_ok=True)
os.makedirs("precomputed", exist_ok=True)
os.makedirs("pretrained", exist_ok=True)
os.makedirs("runs", exist_ok=True)
os.makedirs("runs/guided", exist_ok=True)
sys.path.append("./")


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

parser = argparse.ArgumentParser(description='.')
parser.add_argument('img_index', metavar='N', type=int, nargs=1,
                    help='index of the image')
args = parser.parse_args()

device = 'cuda'
model_path = os.path.join("pretrained/celeba_hq.ckpt")

exp_dir = f"runs/guided"
os.makedirs(exp_dir, exist_ok=True)

def get_scheduler(s):
    scheduler = set()
    periods = s.split(",")
    for p in periods:
        l = int(p.split("-")[0])
        r = int(p.split("-")[1])
        for i in range(l, r):
            scheduler.add(i)
    return scheduler


img_index = args.img_index[0]
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
    'bs_test': 1, 
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



with open("runs/interpolation/data_1.obj","rb") as f:
    data_list = pickle.load(f)


device = torch.device("cuda")
config.device = device
runner = OurDDPM(args, config, device=device)
runner.load_classifier("checkpoint/attr_classifier_4_attrs_40.pt", 4)

ATTR = 1

res_list = []
classifier_score = []

# for i, scale in enumerate([-200, -100, 0, 100, 200]):
for i, scale in enumerate([0, 100, 200, 300, 400]):
# for i, scale in enumerate([0]):
# for i, guidance_start in enumerate([0, 200, 400, 600, 800]):

    g_sch = get_scheduler(f"0-999")
    runner.args.guidance_scheduler = g_sch

    xt = data_list[img_index]["xt"]
    noise_traj = torch.tensor(data_list[img_index]["noise_traj"]).cuda()
    img = runner.guided_generate_ddpm(xt, var_scheduler, runner.classifier, 1, classifier_scale=scale, noise_traj=noise_traj, attr=ATTR)
    res_list.append(img)
    t = (torch.ones(1) * 0).cuda()
    classifier_score.append(F.sigmoid(runner.classifier(img.cuda(), t)[:, ATTR])[0].item())
    torch.cuda.empty_cache()

res = fuse(res_list)
cv2.imwrite(os.path.join(exp_dir, f'guided_smile_scale{scale}_{img_index}.png'), res)
# res = Image.fromarray(res)
# res.save(os.path.join(exp_dir, f'guided_person_{img_index}.png'))
# tvu.save_image(res, os.path.join(exp_dir, f'guided_person_{img_index}.png'))
print(classifier_score)
