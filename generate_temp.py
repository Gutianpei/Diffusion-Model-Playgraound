import os

from ourddpm import OurDDPM
from main import dict2namespace
import argparse
import yaml
from PIL import Image
import warnings
from deepface import DeepFace
import torch
import pdb
import cv2
import glob
import pickle
import torchvision.utils as tvu

from models.improved_ddpm.unet import EncoderUNetModel

warnings.filterwarnings(action='ignore')

def concat(img_list):
    return cv2.hconcat(img_list)


def create_classifier(
    image_size = 256,
    classifier_use_fp16 = False,
    classifier_width = 256,
    classifier_depth = 1,
    classifier_attention_resolutions = "16,8,4",
    classifier_use_scale_shift_norm = True,
    classifier_resblock_updown = True,
    classifier_pool = "spatial_v2",
):
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=classifier_width,
        out_channels=1,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
    )

device = 'cuda'

model_path = os.path.join("pretrained/celeba_hq.ckpt")




exp_dir = f"runs/attrs_glass_gender"
os.makedirs(exp_dir, exist_ok=True)

n_step =  999#@param {type: "integer"}
sampling = "ddpm" #@param ["ddpm", "ddim"]
fixed_xt = False #@param {type: "boolean"}
add_var = True #@param {type: "boolean"}
add_var_on = "0-999" #@param {type: "string"}
vis_gen =  False #@param {type: "boolean"}

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

if bool(add_var):
    var_scheduler = []
    periods = add_var_on.split(",")
    for period in periods:
        start = int(period.split("-")[0])
        end = int(period.split("-")[1])
        for n in range(start,end):
            var_scheduler.append(n)



runner = OurDDPM(args, config, device=device)
# create 50 data point

device = torch.device("cuda")
if bool(fixed_xt):
    torch.manual_seed(0)
    xt = torch.randn((1,3,256,256),device=device)
else:
    xt = torch.randn((1,3,256,256),device=device)
config.device = device

classifier_glass = create_classifier()
classifier_gender = create_classifier()
classifier_path1 = os.path.join(f"glass_classifier_sgd_best.pt")
classifier_glass.load_state_dict(torch.load(classifier_path1))
classifier_path2 = os.path.join(f"gender_classifier_sgd_best.pt")
classifier_gender.load_state_dict(torch.load(classifier_path2))
classifier = [classifier_glass.cuda().eval(), classifier_gender.cuda().eval()]

img_list = []


for scale in [20, 50, 70, 100]:
    for i in range(4):
        img_list = []
        for i in range(4):
            print(f"Generate scale {scale} sample {i}")
            img = runner.guided_generate_ddpm(xt,var_scheduler, classifier, scale)
            img = img[0].permute(1,2,0).detach().cpu().numpy()

            img_list.append(img)
        concat_img = cv2.hconcat(img_list) * 255
        #pdb.set_trace()
        img_name = f"glass_gender_{scale:03d}.png"
        img_path = os.path.join(args.image_folder, img_name)
        cv2.imwrite(img_path, cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR))
