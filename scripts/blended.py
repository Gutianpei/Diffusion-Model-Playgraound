from collections import defaultdict
import os,sys
sys.path.append("./")

from datetime import datetime
from collections import OrderedDict
from ourddpm import OurDDPM
from models.ddpm.diffusion import DDPM
from utils.image_utils import fuse, normalize
from utils.diffusion_utils import clip_normalize
from main import dict2namespace
import argparse
import yaml
import warnings
warnings.filterwarnings(action='ignore')
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import torchvision.utils as tvu
import torch.nn.functional as F
import clip
import lpips

# command line arguments
parser = argparse.ArgumentParser(description='.')
parser.add_argument('img_index', metavar='N', type=int,
                    help='index of the image')
parser.add_argument("-g", "--guidance_on", default=["0-999"], nargs="*", help='steps on which to apply gradient guidance')
parser.add_argument("-m", "--guidance_mask", type=str, default=None, help="location of the guidance mask")
parser.add_argument("-c", "--guidance_model_checkpoint", type=str, default="checkpoint/attr_classifier_4_attrs_150.pt", help="location of the guidance model checkpoint")
parser.add_argument("-r", "--restoration_model_checkpoint", type=str, default="checkpoint/epoch_180.pt", help="restoration checkpoint location")
parser.add_argument("-t", "--guidance_type", type=str, 
    choices=["clip", 
             "classifier", 
             "directional-clip-loss", 
             "blended", 
             "restoration_blended", 
             "restoration_lpips",
             "restoration_classifier"], default="classifier", help="guidance type")
parser.add_argument("-a", "--attribute", type=int, default=0, help="attribute to guide")
parser.add_argument("-s", "--scale", default=[10], nargs="*", help="guidance scale(s)")
parser.add_argument("-l", "--log", action='store_true', default=False, help="whether or not to store various data in output_dict")
parser.add_argument("-d", "--exp_dir", default="runs/guided", help="output directory")
parser.add_argument("-k", "--keep_att_scores", action='store_true', default=False, help="whether or not to keep the score of the other attributes constant")
parser.add_argument("-C", "--CAM", action='store_true', default=False, help="use gradCAM mask")
input_args = parser.parse_args()
print(input_args)

now = datetime.now()
current_time = now.strftime("%m%d%H%M%S")

# make folders
os.makedirs("checkpoint", exist_ok=True)
os.makedirs("precomputed", exist_ok=True)
os.makedirs("pretrained", exist_ok=True)
os.makedirs("runs", exist_ok=True)
os.makedirs("runs/guided", exist_ok=True)
os.makedirs("runs/plots", exist_ok=True)
os.makedirs(input_args.exp_dir, exist_ok=True)


device = 'cuda'
model_path = os.path.join("pretrained/celeba_hq.ckpt")
exp_dir = input_args.exp_dir
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

img_index = input_args.img_index
n_step =  999#@param {type: "integer"}
sampling = "ddpm" #@param ["ddpm", "ddim"]
fixed_xt = True #@param {type: "boolean"}
add_var = True #@param {type: "boolean"}
add_var_on = "0-999" #@param {type: "string"}
vis_gen =  True #@param {type: "boolean"}
var_scheduler = get_scheduler(add_var_on)
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
    }
args = dict2namespace(args_dic)

with open(os.path.join('configs', args.config), 'r') as f:
    config_dic = yaml.safe_load(f)
config = dict2namespace(config_dic)

ATTRIBUTES=['makeup', 'smile', 'glasses', 'gender']
def draw_figure(arrs, title, path):
    plt.figure()
    for i, a in enumerate(arrs):
        plt.plot(a, label=ATTRIBUTES[i])
    plt.title(title)
    plt.legend()
    plt.savefig(path)

with open("runs/interpolation/data_1.obj","rb") as f:
    data_list = pickle.load(f)

guidance_mask = None
if input_args.guidance_mask is not None:
    guidance_mask = np.array(Image.open(input_args.guidance_mask))
    guidance_mask = torch.tensor(guidance_mask).float().cuda() / 255

# load guidance model
device = torch.device("cuda")
config.device = device
runner = OurDDPM(args, config, device=device)
runner.load_classifier(input_args.guidance_model_checkpoint, 4)
if input_args.guidance_type == "classifier":
    guidance_model = runner.classifier
    clip_model = None
else:
    # load restoration network
    guidance_model = DDPM(config)
    ckpt = torch.load(input_args.restoration_model_checkpoint)
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    guidance_model.load_state_dict(new_state_dict)
    guidance_model.to(device)
    runner.restoration_model = guidance_model

    if "restoration" in input_args.guidance_type:
        class Temp(torch.nn.Module):
            def __init__(self, a, b):
                super().__init__()
                self.restore = a
                self.classify = b
            def forward(self, x, t):
                x = self.a(x, t)
                x = self.b(x, torch.zeros_like(t).cuda())
                return x

        guidance_model = Temp(guidance_model, runner.classifier)
        clip_model = None
        if input_args.guidance_type == "restoration_classifier":
            input_args.guidance_type = "classifier"
    else:
        # load CLIP model
        clip_model, preprocess = clip.load("RN101", device=device)

if "lpips" in input_args.guidance_type:
    lpips_model = lpips.LPIPS(net='vgg').cuda()
else:
    lpips_model = None

if "clip" not in input_args.guidance_type:
    clip_target_text = None
    clip_target_img = None
else:
    clip_target_text = ["a woman with glasses"]
    clip_reference_text = ["a woman"]
    clip_target_text = clip.tokenize(clip_target_text).to(device)
    clip_target_text = clip_model.encode_text(clip_target_text)  # (1, 512)
    clip_reference_text = clip.tokenize(clip_reference_text).to(device)
    clip_reference_text = clip_model.encode_text(clip_reference_text)  # (1, 512)
    clip_direction = clip_target_text - clip_reference_text

original_img_clip = None
target_img_clip = None

scales = [int(e) for e in input_args.scale]
guidance_on  = input_args.guidance_on


x_orig = None
res_list = []

# x_orig = torch.tensor((np.array(Image.open("runs/original/tony.jpg").resize((256,256))))).permute(2, 0, 1).cuda().float().unsqueeze(0)/255 * 2 - 1
# guidance_mask = np.array(Image.open("runs/masks/glasses/tony.jpg"))
# guidance_mask = torch.tensor(guidance_mask).float().cuda() / 255

# res_list = [x_orig.cpu()]
classifier_score = []

for j, guidance in enumerate(guidance_on):
    for i, scale in enumerate(scales):
        output_dict = None
        if input_args.log:
            output_dict = defaultdict(list)

        # set guidance scheduler
        g_sch = get_scheduler(guidance)
        runner.args.guidance_scheduler = g_sch
        # get noise trajectory
        xt = data_list[img_index]["xt"]
        noise_traj = torch.tensor(data_list[img_index]["noise_traj"]).cuda()
        
        target_logits = []
        if i != 0 and input_args.keep_att_scores:
            target_logits = [e for e in classifier_score[0]]
            target_logits[input_args.attribute] = 1
        
        img, traj = runner.guided_generate_ddpm(xt, 
                            var_scheduler, 
                            guidance_model, 
                            1, 
                            guidance_scale = scale, 
                            noise_traj = noise_traj, 
                            attr_num = 4,
                            attr = input_args.attribute, 
                            guidance_mask = guidance_mask, 
                            target_logits = target_logits,
                            output_dict = output_dict,
                            clip_target_text = clip_target_text,
                            clip_target_img = target_img_clip,
                            clip_model = clip_model,
                            guidance_mode = input_args.guidance_type,
                            CAM_mask = input_args.CAM,
                            x_orig=x_orig,
                            lpips_model=lpips_model,
                            out_x0_t=True)

        if i == 0 and ("blended" in input_args.guidance_type or "lpips" in input_args.guidance_type):
            x_orig = img.cuda()

        res_list.append(img)
        t = torch.zeros(1).cuda()
        classifier_score.append(F.sigmoid(runner.classifier(img.cuda(), t))[0].tolist())
        if scale == 0 and input_args.guidance_type == 'directional-clip-loss':
            original_img_clip = clip_model.encode_image(clip_normalize(img.cuda()))
            target_img_clip = original_img_clip + clip_direction

        torch.cuda.empty_cache()

        # logits = [output_dict["clip_logit"]]
        # draw_figure(logits, "logits at different time steps", f'runs/plots/{current_time}_{ATTRIBUTES[input_args.attribute]}_{input_args.img_index}_plot.png')
        # norms = [output_dict["attr0_gradient_norm"], output_dict["attr1_gradient_norm"], output_dict["attr2_gradient_norm"], output_dict["attr3_gradient_norm"]]
        # draw_figure(norms, "gradient norms at different time steps", f"runs/plots/norms_{img_index}_{scale}.png")
            

res = fuse(res_list)
cv2.imwrite(os.path.join(exp_dir, f'{current_time}_{ATTRIBUTES[input_args.attribute]}_{input_args.img_index}.png'), res)
for i, e in enumerate(classifier_score):
    print(e)

res = fuse(traj)
cv2.imwrite(os.path.join(exp_dir, f'{current_time}_{ATTRIBUTES[input_args.attribute]}_{input_args.img_index}_traj.png'), res)




