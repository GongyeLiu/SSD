from collections import OrderedDict
from skimage.metrics import structural_similarity as ssim_calc
from skimage.metrics import peak_signal_noise_ratio as psnr_calc
import lpips
import torch
import numpy as np

from trainer.utils import tensor2img
from trainer.metrics.FID import load_patched_inception_v3, get_inception_features_mu_sigma, calculate_fid


def img2tensor(img, bgr2rgb=True):
    if bgr2rgb:
        img = img[:, :, ::-1]
    img = img.astype('float32')
    tensor = torch.from_numpy(img.transpose(2, 0, 1))
    tensor = tensor / 255.0
    tensor = tensor.unsqueeze(0)
    return tensor


class ImageMetrics(object):
    def __init__(self, device=None):
        self.inception = load_patched_inception_v3(device)
        self.lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
        self.device = device

    def calc_metrics(self, tensor1=None, tensor2=None, img1=None, img2=None, metrics=('psnr', 'ssim')):
        if img1 is None and img2 is None:
            img1 = tensor2img(tensor1)
            img2 = tensor2img(tensor2)
        if tensor1 is None and tensor2 is None:
            tensor1 = img2tensor(img1).cuda()
            tensor2 = img2tensor(img2).cuda()

        metrics_dict = OrderedDict()
        if 'PSNR' in metrics or "psnr" in metrics:
            metrics_dict['PSNR'] = psnr_calc(img1, img2, data_range=255.0)
        if 'SSIM' in metrics or "ssim" in metrics:
            metrics_dict['SSIM'] = ssim_calc(img1, img2, data_range=255.0, multichannel=True)
        if 'FID' in metrics or "fid" in metrics:
            metrics_dict['FID'] = self.fid_calc(tensor1, tensor2)
        if 'LPIPS' in metrics or 'lpips' in metrics:
            metrics_dict['LPIPS'] = self.lpips_calc(tensor1, tensor2)
        return metrics_dict

    def lpips_calc(self, tensor1, tensor2):
        # normalize to [-1, 1]
        tensor1 = tensor1 * 2 - 1
        tensor2 = tensor2 * 2 - 1
        lpips = self.lpips_loss_fn(tensor1, tensor2)
        return lpips.item()

    def fid_calc(self, tensor1, tensor2):
        # inception model
        mu1, sigma1 = get_inception_features_mu_sigma(tensor1, self.inception)
        mu2, sigma2 = get_inception_features_mu_sigma(tensor2, self.inception)
        fid = calculate_fid(mu1, sigma1, mu2, sigma2)
        return fid



