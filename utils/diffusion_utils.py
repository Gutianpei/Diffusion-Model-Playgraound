import pickle
import numpy as np
import torch
import pdb
import torch.nn.functional as F
import clip
import torchvision.transforms as tfs
import pdb
from .cam_utils import gradCAM
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
import copy

def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out

#newly added argument: add_var
# target_logits: an array specifying the logit values we want for each attribute.
# when this is specified, the scale should be positive
# example: [1, 0.4, 0.6, 0.2]. we want to push attr 0 to 1, attr 1 to 0.4, attr 2 to ...
def denoising_step(xt, t, t_next, *,
                   models,
                   logvars,
                   b,
                   sampling_type='ddpm',
                   eta=0.0,
                   learn_sigma=False,
                   hybrid=False,
                   hybrid_config=None,
                   ratio=1.0,
                   out_x0_t=False,
                   add_var= False,
                   add_var_on = None,
                   guidance_model = None,
                   guidance_scale = None,
                   guidance = False,
                   variance = None,
                   zt = None,
                   attr_num = 4,
                   attr = 0,
                   guidance_mask = None,
                   target_logits = [],
                   output_dict = None,
                   guidance_mode = "classifier",
                   clip_target_text = None,
                   clip_target_img = None,
                   clip_model = None,
                   CAM_mask = False,
                   x_orig = None,
                   lpips_model = None,
                   restoration_model = None,
                   ):

    # Compute noise and variance
    if type(models) != list:
        model = models.diffusion_model
        et = model(xt, t)
        if learn_sigma:
            et, logvar_learned = torch.split(et, et.shape[1] // 2, dim=1)
            logvar = logvar_learned
        else:
            logvar = extract(logvars, t, xt.shape)
        # x0_t = models.predict_xstart_from_eps(xt, t, et)
        # x0_t = guidance_model.restore(xt, t)
        # x0_t = xt
        
    else:
        if not hybrid:
            et = 0
            logvar = 0
            if ratio != 0.0:
                et_i = ratio * models[1](xt, t)
                if learn_sigma:
                    et_i, logvar_learned = torch.split(et_i, et_i.shape[1] // 2, dim=1)
                    logvar += logvar_learned
                else:
                    logvar += ratio * extract(logvars, t, xt.shape)
                et += et_i

            if ratio != 1.0:
                et_i = (1 - ratio) * models[0](xt, t)
                if learn_sigma:
                    et_i, logvar_learned = torch.split(et_i, et_i.shape[1] // 2, dim=1)
                    logvar += logvar_learned
                else:
                    logvar += (1 - ratio) * extract(logvars, t, xt.shape)
                et += et_i

        else:
            for thr in list(hybrid_config.keys()):
                if t.item() >= thr:
                    et = 0
                    logvar = 0
                    for i, ratio in enumerate(hybrid_config[thr]):
                        ratio /= sum(hybrid_config[thr])
                        et_i = models[i+1](xt, t)
                        if learn_sigma:
                            et_i, logvar_learned = torch.split(et_i, et_i.shape[1] // 2, dim=1)
                            logvar_i = logvar_learned
                        else:
                            logvar_i = extract(logvars, t, xt.shape)
                        et += ratio * et_i
                        logvar += ratio * logvar_i
                    break
    if variance is not None:
        var = extract(variance, t, xt.shape)
    # Compute the next x
    bt = extract(b, t, xt.shape)
    at = extract((1.0 - b).cumprod(dim=0), t, xt.shape)

    if t_next.sum() == -t_next.shape[0]:
        at_next = torch.ones_like(at)
    else:
        at_next = extract((1.0 - b).cumprod(dim=0), t_next, xt.shape)

    xt_next = torch.zeros_like(xt)
    if sampling_type == 'ddpm':
        weight = bt / torch.sqrt(1 - at)

        mean = 1 / torch.sqrt(1.0 - bt) * (xt - weight * et)

        #add_var: variance sigma_t*z is added during the sampling process

        # classifier guided genetation
        if guidance_mode == "classifier" and (guidance_model is not None) and guidance and guidance_scale != 0:
            with torch.enable_grad():
                if len(target_logits) == 0:
                    gradient = get_classifier_gradient(guidance_model, xt, t, attr, 1, output_dict)
                else:
                    gradients = []
                    for i in range(attr_num):
                        gradient = get_classifier_gradient(guidance_model, xt, t, i, target_logits[i], output_dict)
                        gradients.append(gradient)
                    gradient = torch.zeros_like(xt)
                    for i, each in enumerate(gradients):
                        # gradient = (gradient + each) if i in [0, 3] else gradient
                        gradient = gradient + each

                # gradient = gradient / torch.linalg.norm(gradient, dim=(2, 3)).reshape(-1, 3, 1, 1)
                gradient = -gradient        # gradient descent here

            gradient = gradient * guidance_scale
            if guidance_mask is not None:
                gradient = gradient * guidance_mask
            mean = mean.float() + var * gradient.float()
        
        # clip guided generation
        elif guidance_mode == "clip" and (guidance_model is not None) and guidance and guidance_scale != 0:
            assert clip_target_text is not None, "clip_target_text cannot be None"
            assert clip_model is not None, "clip_model cannot be None"
            with torch.enable_grad():
                x_in = xt.detach().requires_grad_(True)
                img = guidance_model(x_in, t)
                x_restored = clip_normalize(img)
                x_restored = clip_model.encode_image(x_restored)
                x_restored = F.normalize(x_restored, dim=-1)                # (1, 512), unit row vector
                clip_target_text = F.normalize(clip_target_text, dim=-1)    # (1, 512), unit row vector
                dot_product = (x_restored @ clip_target_text.T).squeeze()
                logit = dot_product

                if output_dict is not None:
                    output_dict['clip_logit'].append(logit.detach().cpu().item())
                gradient = torch.autograd.grad(logit, x_in)[0]

                if CAM_mask:
                    # x_in = xt.detach().requires_grad_(True)
                    img = img.detach().requires_grad_(True)
                    with open("temp.pkl", "wb") as f:
                        pickle.dump(img, f)
                    exit(0)
                    # CAM_model = Restoration_at_t(guidance_model, t)
                    # CAM = get_cam(CAM_model, clip_model, x_in, clip_target_text, clip_reference_text)
                    # if output_dict is not None and int(t.item())% 100 == 99:
                    #     output_dict["CAM"].append(CAM)
                    # gradient = gradient * torch.tensor(CAM).cuda()

                    # CAM_model = Clip_encode(clip_model)
                    # CAM = get_cam(CAM_model, clip_model, img, clip_target_text, clip_reference_text)
                    # CAM = get_cam(clip_model.visual, clip_model, clip_normalize(img), clip_target_text, clip_reference_text)

                    CAM = gradCAM(clip_model.visual, clip_normalize(img), clip_target_text, clip_model.visual.layer4)
                    CAM = pad_cam(CAM)
                    # print(CAM.shape)
                    # print(CAM)
                    # exit(0)
                    if output_dict is not None and int(t.item())% 100 == 99:
                        output_dict["CAM"].append(CAM.detach().cpu())
                    gradient = gradient * torch.tensor(CAM).cuda()

            if output_dict is not None:
                gradient_norm = torch.sqrt(torch.sum(gradient * gradient))
                output_dict['clip_gradient'].append(gradient_norm.detach().cpu().item())

            gradient = gradient * guidance_scale
            if guidance_mask is not None:
                gradient = gradient * guidance_mask
            mean = mean.float() + var * gradient.float()

        elif guidance_mode == "directional-clip-loss" and (guidance_model is not None) and guidance and guidance_scale != 0:
            assert clip_target_img is not None, "clip_target_img cannot be None"
            assert clip_model is not None, "clip_model cannot be None"
            with torch.enable_grad():
                x_in = xt.detach().requires_grad_(True)
                img = guidance_model(x_in, t)
                x_restored = clip_normalize(img)
                x_restored = clip_model.encode_image(x_restored)
                x_restored = F.normalize(x_restored, dim=-1)                # (1, 512), unit row vector
                clip_target_img = F.normalize(clip_target_img, dim=-1)    # (1, 512), unit row vector
                # dot_product = (x_restored @ clip_target_img.T).squeeze()
                logit = torch.linalg.norm(clip_target_img-x_restored)
                # logit = dot_product
                if output_dict is not None:
                    output_dict['clip_logit'].append(logit.detach().cpu().item())
                gradient = torch.autograd.grad(logit, x_in)[0]
            gradient = gradient * guidance_scale
            gradient = - gradient
            if guidance_mask is not None:
                gradient = gradient * guidance_mask
            mean = mean.float() + var * gradient.float()
        elif "blended" in guidance_mode and (guidance_model is not None) and guidance and guidance_scale != 0:
            assert guidance_mask is not None, "guidance_mask cannot be None"
            with torch.enable_grad():
                if len(target_logits) == 0:
                    gradient = get_classifier_gradient(guidance_model, xt, t, attr, 1, output_dict)
                else:
                    raise NotImplementedError
            gradient = - gradient
            gradient = gradient * guidance_scale
            mean = mean.float() + var * gradient.float()
            
            if t == 0:
                mean = x_orig * (1 - guidance_mask) + mean * guidance_mask
            else:
                mean = q_sample(b, x_orig, t-1) * (1 - guidance_mask) + mean * guidance_mask 
        elif guidance_mode == "restoration_lpips" and (guidance_model is not None) and guidance and guidance_scale != 0:
            with torch.enable_grad():
                if len(target_logits) == 0:
                    gradient = get_gradient_with_lpips(guidance_model, lpips_model, xt, x_orig, t, attr, 1, output_dict)
                else:
                    raise NotImplementedError
            gradient = - gradient
            gradient = gradient * guidance_scale
            mean = mean.float() + var * gradient.float()
        # elif guidance_mode == "restoration_clip" and (guidance_model is not None) and guidance and guidance_scale != 0:
            

        if add_var and int(t.item()) in add_var_on:
            noise = torch.randn_like(xt) if zt == None else zt
            mask = 1 - (t == 0).float()
            mask = mask.reshape((xt.shape[0],) + (1,) * (len(xt.shape) - 1))
            xt_next = mean + mask * torch.exp(0.5 * logvar) * noise
        else:
            xt_next = mean
        xt_next = xt_next.float()

    elif sampling_type == 'ddim':
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        if eta == 0:
            xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
        elif at > (at_next):
            print('Inversion process is only possible with eta = 0')
            raise ValueError
        else:
            c1 = eta * ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(xt)

    if out_x0_t == True:
        return xt_next, x0_t
    else:
        return xt_next

def get_classifier_gradient(classifier, xt, t, attr, target_logit, output_dict=None):
    x_in = xt.detach().requires_grad_(True)
    output = classifier(x_in, t)
    logits = F.sigmoid(output[:, attr])
    if output_dict is not None:
        output_dict[f"attr{attr}_logit"].append(logits.cpu().detach().item())
    target_logit = max(target_logit, 1e-5)
    target_logit = torch.tensor(target_logit).reshape(logits.shape).cuda()
    log_probs = torch.log(target_logit) - torch.log(logits)
    log_probs = torch.abs(log_probs)                        # L1
    # log_probs = torch.sqrt(log_probs * log_probs)         # L2
    gradient = torch.autograd.grad(log_probs, x_in)[0]
    if output_dict is not None:
        norm = torch.sqrt(torch.sum(torch.tensor(gradient * gradient))).cpu().detach().item()
        output_dict[f"attr{attr}_gradient_norm"].append(norm)
    return gradient

def get_gradient_with_lpips(guidance_model, lpips_model, xt, x_orig, t, attr, target_logit, output_dict):
    x_in = xt.detach().requires_grad_(True)
    restored = guidance_model.restore(x_in, t)
    output = guidance_model.classify(restored, torch.zeros_like(t).cuda())
    logits = F.sigmoid(output[:, attr])
    if output_dict is not None:
        output_dict[f"attr{attr}_logit"].append(logits.cpu().detach().item())
    target_logit = max(target_logit, 1e-5)
    target_logit = torch.tensor(target_logit).reshape(logits.shape).cuda()
    log_probs = torch.log(target_logit) - torch.log(logits)
    log_probs = torch.abs(log_probs)                        # L1
    # log_probs = torch.sqrt(log_probs * log_probs)         # L2
    lpips_loss = lpips_model(restored, x_orig).sum()
    r_loss = range_loss(restored).sum()
    # print(log_probs, lpips_loss)
    loss = 20 * lpips_loss + log_probs + r_loss * 10
    # loss = log_probs
    gradient = torch.autograd.grad(loss, x_in)[0]
    if output_dict is not None:
        norm = torch.sqrt(torch.sum(torch.tensor(gradient * gradient))).cpu().detach().item()
        output_dict[f"attr{attr}_gradient_norm"].append(norm)
    return gradient



def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])

def q_sample(betas, x_start, t, noise=None):
    """
    Diffuse the data for a given number of diffusion steps.
    In other words, sample from q(x_t | x_0).
    :param beta: beta schedule
    :param x_start: the initial data batch.
    :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
    :param noise: if specified, the split-out normal noise.
    :return: A noisy version of x_start.
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    assert noise.shape == x_start.shape

    alphas = 1.0 - betas
    alphas = np.array(alphas.detach().cpu())
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alhpas_cumprod = np.sqrt(1 - alphas_cumprod)

    return (
        extract(sqrt_alphas_cumprod, t, x_start.shape) * x_start
        + extract(sqrt_one_minus_alhpas_cumprod, t, x_start.shape)
        * noise
    )

def clip_normalize(img):
    trans = tfs.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    crop = tfs.CenterCrop(224)
    return trans(crop(img))
def pad_cam(cam):
    '''CAM outputs are 224x224. We need to pad it back to 256x256'''
    pad = tfs.CenterCrop(256)
    return pad(cam)

# class CLIPTarget():
#     def __init__(self, clip_model, target_text, clip_reference_text=None):
#         self.clip_target_text = target_text
#         self.clip_reference_text = clip_reference_text
#         self.clip_model = clip_model

#     def __call__(self, model_output):
#         img = clip_normalize(model_output).unsqueeze(0)
#         # print(img.shape)
#         # exit(0)
#         img = self.clip_model.encode_image(img)
#         img = F.normalize(img, dim=-1)                # (1, 512), unit row vector
#         if self.clip_reference_text is not None:
#             dot_product = (img @ (self.clip_target_text.T - self.clip_reference_text.T)).squeeze()
#         else:
#             dot_product = (img @ self.clip_target_text.T).squeeze()
#         return dot_product

# class CLIPTarget():
#     def __init__(self, target_text, clip_reference_text=None):
#         self.clip_target_text = F.normalize(target_text, dim=-1).to(torch.float32)
#         self.clip_reference_text = clip_reference_text
#         if clip_reference_text is not None:
#             self.clip_reference_text = F.normalize(clip_reference_text, dim=-1).to(torch.float32)

#     def __call__(self, img_embedding):
#         img = F.normalize(img_embedding, dim=-1)
#         if self.clip_reference_text is not None:
#             dot_product = (img @ (self.clip_target_text.T - self.clip_reference_text.T)).squeeze()
#         else:
#             dot_product = (img @ self.clip_target_text.T).squeeze()
#         return dot_product


# class Restoration_at_t(torch.nn.Module):
#     def __init__(self, restoration_model, t):
#         super().__init__()
#         self.model = restoration_model
#         self.t = t
#         self.target_layer = self.model.conv_out
    
#     def forward(self, input):
#         return self.model(input, self.t)

# class Clip_encode(torch.nn.Module):
#     def __init__(self, clip_model):
#         super().__init__()
#         self.model = clip_model.visual
#         self.target_layer = self.model.layer4
    
#     def forward(self, input):
#         return self.model(clip_normalize(input)).to(torch.float32)


# def get_cam(model, clip_model, input_tensor, target_text, reference_text = None):
#     target_layers = [model.target_layer[-1]]
#     # targets = [CLIPTarget(clip_model, target_text, reference_text)]
#     targets = [CLIPTarget(target_text, reference_text)]
#     with GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:
#         pdb.set_trace()
#         grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
#     return grayscale_cam


def denoising_step_return_noise(xt, t, t_next, *,
                   models,
                   logvars,
                   b,
                   eta=0.0,
                   learn_sigma=False,
                   hybrid=False,
                   hybrid_config=None,
                   ratio=1.0,
                   add_var= False,
                   add_var_on = None
                   ):

    # Compute noise and variance
    if type(models) != list:
        model = models
        et = model(xt, t)
        if learn_sigma:
            et, logvar_learned = torch.split(et, et.shape[1] // 2, dim=1)
            logvar = logvar_learned
        else:
            logvar = extract(logvars, t, xt.shape)
    else:
        if not hybrid:
            et = 0
            logvar = 0
            if ratio != 0.0:
                et_i = ratio * models[1](xt, t)
                if learn_sigma:
                    et_i, logvar_learned = torch.split(et_i, et_i.shape[1] // 2, dim=1)
                    logvar += logvar_learned
                else:
                    logvar += ratio * extract(logvars, t, xt.shape)
                et += et_i

            if ratio != 1.0:
                et_i = (1 - ratio) * models[0](xt, t)
                if learn_sigma:
                    et_i, logvar_learned = torch.split(et_i, et_i.shape[1] // 2, dim=1)
                    logvar += logvar_learned
                else:
                    logvar += (1 - ratio) * extract(logvars, t, xt.shape)
                et += et_i

        else:
            for thr in list(hybrid_config.keys()):
                if t.item() >= thr:
                    et = 0
                    logvar = 0
                    for i, ratio in enumerate(hybrid_config[thr]):
                        ratio /= sum(hybrid_config[thr])
                        et_i = models[i+1](xt, t)
                        if learn_sigma:
                            et_i, logvar_learned = torch.split(et_i, et_i.shape[1] // 2, dim=1)
                            logvar_i = logvar_learned
                        else:
                            logvar_i = extract(logvars, t, xt.shape)
                        et += ratio * et_i
                        logvar += ratio * logvar_i
                    break

    # Compute the next x
    bt = extract(b, t, xt.shape)
    at = extract((1.0 - b).cumprod(dim=0), t, xt.shape)

    xt_next = torch.zeros_like(xt)

    weight = bt / torch.sqrt(1 - at)

    mean = 1 / torch.sqrt(1.0 - bt) * (xt - weight * et)

    #add_var: variance sigma_t*z is added during the sampling process
    #   pdb.set_trace()
    if add_var and int(t.item()) in add_var_on:
        noise = torch.randn_like(xt)
        mask = 1 - (t == 0).float()
        mask = mask.reshape((xt.shape[0],) + (1,) * (len(xt.shape) - 1))
        xt_next = mean + mask * torch.exp(0.5 * logvar) * noise
    else:
        xt_next = mean
    xt_next = xt_next.float()

    if add_var and int(t.item()) in add_var_on:
        return xt_next, noise

    return xt_next, 0


def denoising_with_traj(xt, t, t_next, *,
                   models,
                   logvars,
                   b,
                   eta=0.0,
                   learn_sigma=False,
                   hybrid=False,
                   hybrid_config=None,
                   ratio=1.0,
                   add_var= False,
                   add_var_on = None,
                   zt = 0
                   ):

    # Compute noise and variance
    if type(models) != list:
        model = models
        et = model(xt, t)
        if learn_sigma:
            et, logvar_learned = torch.split(et, et.shape[1] // 2, dim=1)
            logvar = logvar_learned
        else:
            logvar = extract(logvars, t, xt.shape)
    else:
        if not hybrid:
            et = 0
            logvar = 0
            if ratio != 0.0:
                et_i = ratio * models[1](xt, t)
                if learn_sigma:
                    et_i, logvar_learned = torch.split(et_i, et_i.shape[1] // 2, dim=1)
                    logvar += logvar_learned
                else:
                    logvar += ratio * extract(logvars, t, xt.shape)
                et += et_i

            if ratio != 1.0:
                et_i = (1 - ratio) * models[0](xt, t)
                if learn_sigma:
                    et_i, logvar_learned = torch.split(et_i, et_i.shape[1] // 2, dim=1)
                    logvar += logvar_learned
                else:
                    logvar += (1 - ratio) * extract(logvars, t, xt.shape)
                et += et_i

        else:
            for thr in list(hybrid_config.keys()):
                if t.item() >= thr:
                    et = 0
                    logvar = 0
                    for i, ratio in enumerate(hybrid_config[thr]):
                        ratio /= sum(hybrid_config[thr])
                        et_i = models[i+1](xt, t)
                        if learn_sigma:
                            et_i, logvar_learned = torch.split(et_i, et_i.shape[1] // 2, dim=1)
                            logvar_i = logvar_learned
                        else:
                            logvar_i = extract(logvars, t, xt.shape)
                        et += ratio * et_i
                        logvar += ratio * logvar_i
                    break

    # Compute the next x
    bt = extract(b, t, xt.shape)
    at = extract((1.0 - b).cumprod(dim=0), t, xt.shape)

    xt_next = torch.zeros_like(xt)

    weight = bt / torch.sqrt(1 - at)

    mean = 1 / torch.sqrt(1.0 - bt) * (xt - weight * et)

    #add_var: variance sigma_t*z is added during the sampling process
    #   pdb.set_trace()
    if add_var and int(t.item()) in add_var_on:
        noise = zt
        mask = 1 - (t == 0).float()
        mask = mask.reshape((xt.shape[0],) + (1,) * (len(xt.shape) - 1))
        xt_next = mean + mask * torch.exp(0.5 * logvar) * noise
    else:
        xt_next = mean
    xt_next = xt_next.float()

    return xt_next, zt
