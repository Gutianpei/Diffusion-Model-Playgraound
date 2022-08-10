from collections import defaultdict
import os,sys
os.makedirs("checkpoint", exist_ok=True)
os.makedirs("precomputed", exist_ok=True)
os.makedirs("pretrained", exist_ok=True)
os.makedirs("runs", exist_ok=True)
os.makedirs("runs/guided", exist_ok=True)
os.makedirs("runs/plots", exist_ok=True)
sys.path.append("./")


from ourddpm import OurDDPM
from models.ddpm.diffusion import DDPM
from utils.image_utils import fuse, normalize
from main import dict2namespace
import argparse
import yaml
import warnings
warnings.filterwarnings(action='ignore')
from PIL import Image
import cv2
import matplotlib.pyplot as plt


import torch
import pickle
import torchvision.utils as tvu
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential
import clip

parser = argparse.ArgumentParser(description='.')
parser.add_argument('img_index', metavar='N', type=int, nargs=1,
                    help='index of the image')
args = parser.parse_args()

device = 'cuda'
model_path = os.path.join("pretrained/celeba_hq.ckpt")

# exp_dir = f"runs/guided"
exp_dir = f"runs/with_restoration"
os.makedirs(exp_dir, exist_ok=True)
ckpt_dir = f"checkpoint"


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

ATTRIBUTES = ["makeup", "smile", "glasses", "gender"]

with open("runs/interpolation/data_1.obj","rb") as f:
    data_list = pickle.load(f)

device = torch.device("cuda")
config.device = device
runner = OurDDPM(args, config, device=device)


def draw_figure(arrs, title, path):
    plt.figure()
    for i, a in enumerate(arrs):
        plt.plot(a, label=ATTRIBUTES[i])
    plt.title(title)
    plt.legend()
    plt.savefig(path)


# LOAD CLASSIFIER
# runner.load_classifier("checkpoint/naive_classifier_4_attrs_30.pt", 4)
# runner.load_classifier("checkpoint/attr_classifier_4_attrs_150.pt", 4)

# LOAD RESTORATION NET
restore_ckpt_dir = os.path.join(ckpt_dir, f"epoch_{180}.pt")
restoration = DDPM(config)
ckpt = torch.load(restore_ckpt_dir)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in ckpt.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

restoration.load_state_dict(new_state_dict)
restoration.to(device)

# class Restore_Classify(nn.Module):
#     def __init__(self, restore, classifier):
#         super().__init__()
#         self.restore = restore
#         self.classifier = classifier
#         self.device = torch.device("cuda")

#     def forward(self, x, t):
#         n = x.shape[0]
#         x = self.restore(x, t)
#         t = (torch.zeros(n)).to(self.device)
#         x = self.classifier(x, t)
#         return x

# restore_classify = Restore_Classify(restoration, runner.classifier)

# load CLIP model
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# INPUT PARAMS
ATTR = 3
target_text = clip.tokenize(["a woman with glasses"]).to(device)
target_text = clip_model.encode_text(target_text)

res_list = []
classifier_score = []

# for i, scale in enumerate([-200, -100, 0, 100, 200]):
# for i, scale in enumerate([-1.2, -0.9, -0.6, -0.3, 0]):
# for i, scale in enumerate([0, 5, 10, 15, 20]):
for i, scale in enumerate([0, 5, 10]):
# for i, guidance in enumerate(["999-999", "750-999", "500-999", "250-999", "0-999"]):
# for i, guidance in enumerate(["999-999", "0-249", "0-499", "0-749", "0-999"]):
# for i, guidance in enumerate(["999-999", "0-599"]):

    g_sch = get_scheduler("0-999")
    # scale = 5
    # g_sch = get_scheduler(guidance)
    runner.args.guidance_scheduler = g_sch

    xt = data_list[img_index]["xt"]
    noise_traj = torch.tensor(data_list[img_index]["noise_traj"]).cuda()
    
    target_logits = []
    # if i != 0:
    #     target_logits = [e for e in classifier_score[0]]
    #     target_logits[ATTR] = 0

    # output_dict = defaultdict(list)
    output_dict = None
    
    img = runner.guided_generate_ddpm(xt, 
                        var_scheduler, 
                        restoration, 
                        1, 
                        guidance_scale = scale, 
                        noise_traj = noise_traj, 
                        attr_num = 4,
                        attr = ATTR, 
                        guidance_mask = None, 
                        target_logits = target_logits,
                        output_dict = output_dict,
                        clip_target_text = None,
                        clip_model = None,
                        guidance_mode = "clip"
                        )
    res_list.append(img)
    t = torch.zeros(1).cuda()
    # classifier_score.append(F.sigmoid(runner.classifier(img.cuda(), t))[0].tolist())
    torch.cuda.empty_cache()

    # logits = [output_dict['clip_logit']]
    # draw_figure(logits, "clip logit at different time steps", f"runs/plots/clip_logits_{img_index}_{scale}.png")
    # gradients = [output_dict['clip_gradient']]
    # draw_figure(gradients, "clip gradient norm at different time steps", f"runs/plots/clip_gradients_{img_index}_{scale}.png")

    

res = fuse(res_list)
cv2.imwrite(os.path.join(exp_dir, f'clip_beard_{img_index}.png'), res)

# for i, e in enumerate(classifier_score):
#     print(e)

# with open(os.path.join(exp_dir, f'clip_beard_{img_index}.log'), "w+") as f:
#     for i, e in enumerate(classifier_score):
#         f.write(str(e))
#         f.write("\n")



