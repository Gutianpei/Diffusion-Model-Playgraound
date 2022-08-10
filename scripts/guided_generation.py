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

ATTRIBUTES = ["makeup", "smile", "glasses", "gender"]

def draw_figure(arrs, title, path):
    plt.figure()
    for i, a in enumerate(arrs):
        plt.plot(a, label=ATTRIBUTES[i])
    plt.title(title)
    plt.legend()
    plt.savefig(path)

with open("runs/interpolation/data_1.obj","rb") as f:
    data_list = pickle.load(f)
# with open("test_mask.pickle", "rb") as f:
#     guidance_mask = torch.tensor(pickle.load(f)).cuda()
# guidance_mask = torch.tensor([[1] * 256 for _ in range(256)]).cuda()

device = torch.device("cuda")
config.device = device
runner = OurDDPM(args, config, device=device)
# runner.load_classifier("checkpoint/naive_classifier_4_attrs_30.pt", 4)
runner.load_classifier("checkpoint/attr_classifier_4_attrs_150.pt", 4)

ATTR = 3

res_list = []
classifier_score = []

# for i, scale in enumerate([-200, -100, 0, 100, 200]):
# for i, scale in enumerate([-1.2, -0.9, -0.6, -0.3, 0]):
for i, scale in enumerate([0, 5, 10]):
# for i, scale in enumerate([-0]):
# for i, guidance in enumerate(["999-999", "750-999", "500-999", "250-999", "0-999"]):
# for i, guidance in enumerate(["999-999", "0-249", "0-499", "0-749", "0-999"]):
# for i, guidance in enumerate(["999-999", "0-499"]):

    g_sch = get_scheduler("0-999")
    # scale = -3
    # g_sch = get_scheduler(guidance)
    runner.args.guidance_scheduler = g_sch

    xt = data_list[img_index]["xt"]
    noise_traj = torch.tensor(data_list[img_index]["noise_traj"]).cuda()
    
    target_logits = []
    # if i != 0:
    #     target_logits = [e for e in classifier_score[0]]
    #     target_logits[ATTR] = 1

    # target_logits = [1, 1, 1, 1]

    # output_dict = defaultdict(list)
    output_dict = None
    
    img = runner.guided_generate_ddpm(xt, 
                        var_scheduler, 
                        runner.classifier, 
                        1, 
                        classifier_scale = scale, 
                        noise_traj = noise_traj, 
                        attr_num = 4,
                        attr = ATTR, 
                        guidance_mask = None, 
                        target_logits = target_logits,
                        output_dict = output_dict)
    res_list.append(img)
    t = torch.zeros(1).cuda()
    classifier_score.append(F.sigmoid(runner.classifier(img.cuda(), t))[0].tolist())
    torch.cuda.empty_cache()

    # logits = [output_dict["attr0_logit"], output_dict["attr1_logit"], output_dict["attr2_logit"], output_dict["attr3_logit"]]
    # draw_figure(logits, "logits at different time steps", f"runs/plots/logits_{img_index}_{scale}.png")
    # norms = [output_dict["attr0_gradient_norm"], output_dict["attr1_gradient_norm"], output_dict["attr2_gradient_norm"], output_dict["attr3_gradient_norm"]]
    # draw_figure(norms, "gradient norms at different time steps", f"runs/plots/norms_{img_index}_{scale}.png")
    


res = fuse(res_list)
cv2.imwrite(os.path.join(exp_dir, f'guided_makeup_{img_index}.png'), res)
for i, e in enumerate(classifier_score):
    print(e)




