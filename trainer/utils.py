import os
import numpy as np
import cv2
import torch
from skimage.metrics import structural_similarity as ssim_calc
from skimage.metrics import peak_signal_noise_ratio as psnr_calc
import importlib


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def normalize(img, mean=0.5, std=0.5, reverse=False):
    if reverse:
        return img * std + mean
    else:
        return (img - mean) / std


def tensor2img(tensor):
    tensor = tensor.detach().cpu().numpy()
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    if tensor.shape[0] == 1:
        tensor = np.tile(tensor, (3, 1, 1))
    tensor = tensor.transpose((1, 2, 0))
    tensor = tensor * 255.0
    tensor = np.clip(tensor, 0, 255)
    tensor = tensor.astype("uint8")
    tensor = tensor[:, :, ::-1]
    return tensor


def calc_psnr_ssim(img1, img2, is_tensor=True):
    if is_tensor:
        img1 = tensor2img(img1)
        img2 = tensor2img(img2)
    psnr = psnr_calc(img1, img2, data_range=255.0)
    ssim = ssim_calc(img1, img2, data_range=255.0, multichannel=True)
    return psnr, ssim


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def calculate_parameters(net):
    out = 0
    for param in net.parameters():
        out += param.numel()
    return out