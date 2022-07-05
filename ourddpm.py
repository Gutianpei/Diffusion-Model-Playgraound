from site import setquit
import time
from glob import glob
from tqdm.auto import tqdm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
import torchvision.utils as tvu
import pdb
import random
from torch.optim import AdamW
import torch.nn.functional as F
import numpy as np


from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from models.improved_ddpm.unet import EncoderUNetModel
from utils.text_dic import SRC_TRG_TXT_DIC
from utils.diffusion_utils import get_beta_schedule, denoising_step, denoising_step_return_noise, denoising_with_traj

from datasets.data_utils import get_dataset, get_dataloader
from configs.paths_config import DATASET_PATHS, MODEL_PATHS, HYBRID_MODEL_PATHS, HYBRID_CONFIG
from datasets.imagenet_dic import IMAGENET_DIC
from utils.align_utils import run_alignment


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

class OurDDPM(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
                             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))


        #newly added
        #self.add_var = self.args.add_var
       # self.add_var_on = self.args.add_var_on

    def train_classifier(self):
        data_root = "/home/guangyi.chen/workspace/gutianpei/diffusion/DMP_data/data/celeba_hq"
        #data_root = "/home/summertony717/data/celeba_hq"
        self.model = create_classifier().to("cuda")
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
        #model.to("cuda")
        #self.model = model
        print("Classifier created")
        train_dataset, val_dataset, test_dataset = get_dataset("CelebA_HQ", data_root, self.config)#, label = "../DMP_data/list_attr_celeba.csv.zip")
        loader_dic = get_dataloader(train_dataset, test_dataset, bs_train=self.args.bs_train,
                                    num_workers=self.config.data.num_workers)
        self.train_loader = loader_dic["train"]
        self.test_loader = loader_dic["test"]

        opt = torch.optim.SGD(self.model.parameters(), lr=0.001)
        #opt = torch.optim.Adam(self.model.parameters(), lr=1se-4)
        criterion = torch.nn.BCELoss()

        self.model.train()
        loss_ls = []
        acc_ls = {}
        for epoch in range(10):
            print(f"Epoch {epoch}")
            # train
            cnt = 0
            with tqdm(total=len(self.train_loader), ncols=80) as pbar:
            #pbar = tqdm(self.train_loader, ncols=80)
            #pbar = self.train_loader
            #pdb.set_trace()
                for index, (img, attrs) in enumerate(self.train_loader):
                    # print(step, flush=True)
                    #pdb.set_trace()
                    N = img.shape[0]
                    cnt += N
                    #pdb.set_trace()
                    label = (attrs[1].reshape(N, 1).float().to("cuda") + 1) / 2     # reshape and normalize
                    x0 = img.to("cuda")
                    # e = torch.randn_like(x0)
                    # a = (1 - self.betas).cumprod(dim=0)
                    # t0 = random.randint(1, 100)
                    # x = x0 * a[t0 - 1].sqrt() + e * (1.0 - a[t0 - 1]).sqrt()
                    # ts = (torch.ones(N) * t0).to("cuda")

                    # testing the code without the randomness first
                    x = x0
                    ts = (torch.ones(N) * 0).to("cuda")

                    #pdb.set_trace()

                    logits = F.sigmoid(self.model(x, timesteps=ts))

                    loss = criterion(logits, label)
                    # loss_ls.append(loss.detach().cpu())
                    pbar.set_description(f"Epoch {epoch}, Loss: {loss.item():.2f}")
                    pbar.update(1)
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                #pdb.set_trace()
                #torch.cuda.empty_cache()
                #if step % 1000 == 0 and step != 0:  # every 100 samples
            self.model.eval()
            print("Evaluating...")
            acc, val_loss = self.eval_classifier(self.test_loader)
            print(f"Epoch {epoch}: validation loss = {val_loss}; acc = {acc}")
            acc_ls[epoch] = acc
            self.model.train()
        print(acc_ls)

            # eval
            # log_probs = F.log_softmax(logits, dim=-1)
            # selected = log_probs[range(len(logits)), y.view(-1)]
            # correct = (log_probs.argmax(dim=1).cpu().numpy() == y.cpu().numpy()).sum()

    def eval_classifier(self, dataset_loader):
        corrects = 0.0
        loss_ls = []
        with torch.no_grad():
            for step, (img, attrs) in enumerate(dataset_loader):
                # print("eval step", step, flush=True)
                N = img.shape[0]
                label_cpu = (attrs[1].reshape(len(attrs[1]), 1).float() + 1) / 2
                label = label_cpu.to("cuda")     # reshape and normalize
                x0 = img.to("cuda")
                # e = torch.randn_like(x0)
                # a = (1 - self.betas).cumprod(dim=0)
                # t0 = random.randint(1, 100)
                # x = x0 * a[t0 - 1].sqrt() + e * (1.0 - a[t0 - 1]).sqrt()
                # ts = (torch.ones(N) * t0).to("cuda")

                # testing the code without the randomness first
                x = x0
                ts = (torch.ones(N) * 0).to("cuda")

                logits = F.sigmoid(self.model(x, timesteps=ts))
                y_pred_tag = torch.round(logits).detach().cpu()

                correct_results_sum = (y_pred_tag == label_cpu).sum().float()
                corrects += correct_results_sum
                #pdb.set_trace()
                loss = F.binary_cross_entropy(logits, label, reduction="mean").detach().cpu()
                loss_ls.append(loss)
        #pdb.set_trace()
        acc = corrects.item()/len(dataset_loader)
        acc = acc * 100
        return acc, np.mean(loss_ls)




    def generate_ddpm(self, xt, sigma):
        # ----------- random noise -----------#
        n = self.args.bs_test
        x0 = torch.randn((n, 3, self.config.data.image_size,self.config.data.image_size), device=self.config.device)
        trajs = []

        # ----------- Models -----------#



        if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
            model_i = DDPM(self.config)

            ckpt = torch.load(self.args.model_path)
            learn_sigma = False
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
            model_i = i_DDPM(self.config.data.dataset)
            if self.args.model_path:
                ckpt = torch.load(self.args.model_path)
            else:
                ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            learn_sigma = True
        else:
            print('Not implemented dataset')
            raise ValueError
        model_i.load_state_dict(ckpt)
        model_i.to(self.device)
        model_i = torch.nn.DataParallel(model_i)
        model_i.eval()
        print(f"{self.args.model_path} is loaded.")

        with torch.no_grad():
            # ----------- Generative Process -----------#
            print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}, "
                  f" Steps: {self.args.n_step}")

            seq_test = list(range(self.args.n_step))
            print('Uniform skip type')

            seq_test_next = [-1] + list(seq_test[:-1])


            # e = torch.randn_like(x0)
            # a = (1 - self.betas).cumprod(dim=0)
            # x = x0 * a[self.args.t_0 - 1].sqrt() + e * (1.0 - a[self.args.t_0 - 1]).sqrt()
            x = xt

            with tqdm(total=len(seq_test), desc="Generative process") as progress_bar:
                for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                    t = (torch.ones(n) * i).to(self.device)
                    t_next = (torch.ones(n) * j).to(self.device)

                    x = denoising_step(x, t=t, t_next=t_next, models=model_i,
                                        logvars=self.logvar,
                                        sampling_type=self.args.sample_type,
                                        b=self.betas,
                                        eta=self.args.eta,
                                        learn_sigma=learn_sigma,
                                        ratio=1,
                                        hybrid=self.args.hybrid_noise,
                                        hybrid_config=HYBRID_CONFIG,
                                        add_var = self.add_var,
                                        add_var_on = sigma)

                    # added intermediate step vis
                    if i % 100 == 0:
                        trajs.append(x.detach().cpu().numpy())
                        tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                     f'ngen{self.args.n_step}_{i:03d}.png'))
                    progress_bar.update(1)
            #tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
        return trajs    #                                                         f'ngen{self.args.n_step}_final.png'))




    def edit_one_image(self):
        # ----------- Data -----------#
        n = self.args.bs_test

        if self.args.align_face and self.config.data.dataset in ["FFHQ", "CelebA_HQ"]:
            try:
                img = run_alignment(self.args.img_path, output_size=self.config.data.image_size)
            except:
                img = Image.open(self.args.img_path).convert("RGB")
        else:
            img = Image.open(self.args.img_path).convert("RGB")
        img = img.resize((self.config.data.image_size, self.config.data.image_size), Image.ANTIALIAS)
        img = np.array(img)/255
        img = torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(dim=0).repeat(n, 1, 1, 1)
        img = img.to(self.config.device)
        tvu.save_image(img, os.path.join(self.args.image_folder, f'0_orig.png'))
        x0 = (img - 0.5) * 2.

        # ----------- Models -----------#
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
            pass
        else:
            raise ValueError

        models = []

        if self.args.hybrid_noise:
            model_paths = [None] + HYBRID_MODEL_PATHS
        else:
            model_paths = [None, self.args.model_path]

        for model_path in model_paths:
            if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
                model_i = DDPM(self.config)
                if model_path:
                    ckpt = torch.load(model_path)
                else:
                    ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
                learn_sigma = False
            elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
                model_i = i_DDPM(self.config.data.dataset)
                if model_path:
                    ckpt = torch.load(model_path)
                else:
                    ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
                learn_sigma = True
            else:
                print('Not implemented dataset')
                raise ValueError
            model_i.load_state_dict(ckpt)
            model_i.to(self.device)
            model_i = torch.nn.DataParallel(model_i)
            model_i.eval()
            print(f"{model_path} is loaded.")
            models.append(model_i)

        with torch.no_grad():
            #---------------- Invert Image to Latent in case of Deterministic Inversion process -------------------#
            if self.args.deterministic_inv:
                x_lat_path = os.path.join(self.args.image_folder, f'x_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}.pth')
                if not os.path.exists(x_lat_path):
                    seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
                    seq_inv = [int(s) for s in list(seq_inv)]
                    seq_inv_next = [-1] + list(seq_inv[:-1])

                    x = x0.clone()
                    with tqdm(total=len(seq_inv), desc=f"Inversion process ") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_prev, models=models,
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=learn_sigma,
                                               ratio=0,
                                               add_var = self.add_var
                                               )

                            progress_bar.update(1)
                        x_lat = x.clone()
                        torch.save(x_lat, x_lat_path)
                else:
                    print('Latent exists.')
                    x_lat = torch.load(x_lat_path)


            # ----------- Generative Process -----------#
            print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}, "
                  f" Steps: {self.args.n_test_step}/{self.args.t_0}")
            if self.args.n_test_step != 0:
                seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
                seq_test = [int(s) for s in list(seq_test)]
                print('Uniform skip type')
            else:
                seq_test = list(range(self.args.t_0))
                print('No skip')
            seq_test_next = [-1] + list(seq_test[:-1])

            for it in range(self.args.n_iter):
                if self.args.deterministic_inv:
                    x = x_lat.clone()
                else:
                    e = torch.randn_like(x0)
                    a = (1 - self.betas).cumprod(dim=0)
                    x = x0 * a[self.args.t_0 - 1].sqrt() + e * (1.0 - a[self.args.t_0 - 1]).sqrt()
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'1_lat_ninv{self.args.n_inv_step}.png'))

                with tqdm(total=len(seq_test), desc="Generative process {}".format(it)) as progress_bar:
                    for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                        t = (torch.ones(n) * i).to(self.device)
                        t_next = (torch.ones(n) * j).to(self.device)

                        x = denoising_step(x, t=t, t_next=t_next, models=models,
                                           logvars=self.logvar,
                                           sampling_type=self.args.sample_type,
                                           b=self.betas,
                                           eta=self.args.eta,
                                           learn_sigma=learn_sigma,
                                           ratio=self.args.model_ratio,
                                           hybrid=self.args.hybrid_noise,
                                           hybrid_config=HYBRID_CONFIG,
                                           add_var = self.add_var)

                        # added intermediate step vis
                        if (i - 99) % 100 == 0:

                            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                       f'2_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_{i}_it{it}.png'))
                        progress_bar.update(1)

                x0 = x.clone()
                if self.args.model_path:
                    tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                               f"3_gen_t{self.args.t_0}_it{it}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_mrat{self.args.model_ratio}_{self.args.model_path.split('/')[-1].replace('.pth','')}.png"))
                else:
                    tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'3_gen_t{self.args.t_0}_it{it}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_mrat{self.args.model_ratio}.png'))


    # modes: gen. generate noise trajectory
    #        use. use noise trajectory: TODO
    def generate_ddpm_and_noise_traj(self, xt, sigma, mode="gen", noise_traj=None):
        # ----------- random noise -----------#
        n = self.args.bs_test
        trajs = []
        if mode == "gen":
            noise_traj = []

        # ----------- Models -----------#
        if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
            model_i = DDPM(self.config)

            ckpt = torch.load(self.args.model_path)
            learn_sigma = False
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
            model_i = i_DDPM(self.config.data.dataset)
            if self.args.model_path:
                ckpt = torch.load(self.args.model_path)
            else:
                ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            learn_sigma = True
        else:
            print('Not implemented dataset')
            raise ValueError
        model_i.load_state_dict(ckpt)
        model_i.to(self.device)
        model_i = torch.nn.DataParallel(model_i)
        model_i.eval()
        print(f"{self.args.model_path} is loaded.")

        with torch.no_grad():
            # ----------- Generative Process -----------#
            print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}, "
                  f" Steps: {self.args.n_step}")
            seq_test = list(range(self.args.n_step))
            # seq_test = np.linspace(0, 1, self.args.n_step) * self.args.n_step
            # seq_test = [int(s) for s in list(seq_test)]
            print('Uniform skip type')

            seq_test_next = [-1] + list(seq_test[:-1])

            x = xt
            with tqdm(total=len(seq_test), desc="Generative process") as progress_bar:
                for w, test in enumerate(zip(reversed(seq_test), reversed(seq_test_next))):
                    i, j = test
                    t = (torch.ones(n) * i).to(self.device)
                    t_next = (torch.ones(n) * j).to(self.device)

                    if mode == "gen":
                        x, noise = denoising_step_return_noise(x, t=t, t_next=t_next, models=model_i,
                                        logvars=self.logvar,
                                        b=self.betas,
                                        eta=self.args.eta,
                                        learn_sigma=learn_sigma,
                                        ratio=1,
                                        hybrid=self.args.hybrid_noise,
                                        hybrid_config=HYBRID_CONFIG,
                                        add_var = self.add_var,
                                        add_var_on = sigma)
                        noise_traj.append(noise.detach().cpu().numpy())
                    elif mode == "use":
                        assert noise_traj is not None, "noise_traj is required for use mode"
                        zt = torch.tensor(noise_traj[w]).to(self.device)
                        x, noise = denoising_with_traj(x, t=t, t_next=t_next, models=model_i,
                                        logvars=self.logvar,
                                        b=self.betas,
                                        eta=self.args.eta,
                                        learn_sigma=learn_sigma,
                                        ratio=1,
                                        hybrid=self.args.hybrid_noise,
                                        hybrid_config=HYBRID_CONFIG,
                                        add_var = self.add_var,
                                        add_var_on = sigma,
                                        zt=zt)
                    # added intermediate step vis
                    if i % 100 == 0:
                        trajs.append(x.detach().cpu().numpy())
                        tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                     f'ngen{self.args.n_step}_{i:03d}.png'))
                    progress_bar.update(1)
        return trajs, noise_traj
