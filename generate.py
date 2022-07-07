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

from models.improved_ddpm.unet import EncoderUNetModel

warnings.filterwarnings(action='ignore')

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
classifier_path = os.path.join("glass_classifier_sgd_best.pt")

classifier = create_classifier()
classifier.load_state_dict(torch.load(classifier_path))

exp_dir = f"runs/raw_ddpm"
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
    'classifier_scale': 2
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

data_dic = {}
device = torch.device("cuda")
if bool(fixed_xt):
    torch.manual_seed(0)
    xt = torch.randn((1,3,256,256),device=device)
else:
    xt = torch.randn((1,3,256,256),device=device)
config.device = device


data_dic["xt"] = xt.detach().cpu().numpy()

for id in range(10):
    runner.guided_generate_ddpm(xt,var_scheduler, classifier, id)



    # fuse output image
    # if bool(vis_gen):
    #     img_dir = sorted(glob.glob(exp_dir+"/*.png"))[::-1]
    #     concat = []
    #     for img in img_dir:
    #         im = cv2.imread(img)
    #         concat.append(im)
    #     concat_img = cv2.hconcat(concat)
    #     display = concat_img
    # else:
    #     img_dir = sorted(glob.glob(exp_dir+"/*.png"))[0]
    #     im = cv2.imread(img_dir)
    #     display = im
#     img_dir = sorted(glob.glob(exp_dir+"/*.png"))[0]
#     # face attribute label
#     attribute = DeepFace.analyze(img_dir, actions = ['gender','age','emotion'], enforce_detection = False, prog_bar = False)
#     age = attribute['age']
#     emotion = attribute['dominant_emotion']
#     gender = attribute["gender"]
#
#     # if int(age) < 20:
#     #     age_label = 1
#     # else:
#     #     age_label = -1
#
#     if emotion == "happy":
#         emotion_label = 1
#     else:
#         emotion_label = -1
#
#     if gender == "Man":
#         gender_label = 1
#     else:
#         gender_label = -1
#
#     data_dic["age"] = int(age)
#     data_dic["emotion"] = emotion_label
#     data_dic["gender"] = gender_label
#     data_list.append(data_dic)
#
#
#     #if (i+1) % 2000 == 0:
# filehandler = open("generated_sample/data_5000.pkl","wb+")
# pickle.dump(data_list,filehandler)
