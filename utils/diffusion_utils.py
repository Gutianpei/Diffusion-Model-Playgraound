import numpy as np
import torch
import pdb
import torch.nn.functional as F
import clip
import torchvision.transforms as tfs
import pdb

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
                   clip_model = None
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
                # x_in = xt.detach().requires_grad_(True)
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
        if guidance_mode == "clip" and (guidance_model is not None) and guidance and guidance_scale != 0:
            assert clip_target_text is not None, "clip_target_text cannot be None"
            assert clip_model is not None, "clip_model cannot be None"
            with torch.enable_grad():
                x_in = xt.detach().requires_grad_(True)
                x_restored = guidance_model(x_in, t)
                x_restored = clip_normalize(x_restored)
                x_restored = clip_model.encode_image(x_restored)
                x_restored = F.normalize(x_restored, dim=-1)                # (1, 512), unit row vector
                clip_target_text = F.normalize(clip_target_text, dim=-1)    # (1, 512), unit row vector
                dot_product = (x_restored @ clip_target_text.T).squeeze()
                logit = dot_product
                if output_dict is not None:
                    output_dict['clip_logit'].append(logit.detach().cpu().item())
                gradient = torch.autograd.grad(logit, x_in)[0]
            gradient = gradient * guidance_scale
            if guidance_mask is not None:
                gradient = gradient * guidance_mask
            if output_dict is not None:
                gradient_norm = torch.sqrt(torch.sum(gradient * gradient))
                output_dict['clip_gradient'].append(gradient_norm.detach().cpu().item())
            mean = mean.float() + var * gradient.float()



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
    norm = torch.sqrt(torch.sum(torch.tensor(gradient * gradient))).cpu().detach().item()
    if output_dict is not None:
        output_dict[f"attr{attr}_gradient_norm"].append(norm)
    return gradient

def clip_normalize(img):
    trans = tfs.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    crop = tfs.CenterCrop(224)
    return trans(crop(img))

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
