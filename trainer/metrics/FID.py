import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from tqdm import tqdm

from trainer.metrics.Inception import InceptionV3


def load_patched_inception_v3(device='cuda', resize_input=True, normalize_input=False):
    # we may not resize the input, but in [rosinality/stylegan2-pytorch] it
    # does resize the input.
    inception = InceptionV3([3], resize_input=resize_input, normalize_input=normalize_input)
    # inception = nn.DataParallel(inception).eval().to(device)
    inception = inception.eval().to(device)
    return inception


@torch.no_grad()
def extract_inception_features(data_generator, inception, len_generator=None):
    """Extract inception features.

    Args:
        data_generator (generator): A data generator.
        inception (nn.Module): Inception model.
        len_generator (int): Length of the data_generator to show the
            progressbar. Default: None.
        device (str): Device. Default: cuda.

    Returns:
        Tensor: Extracted features.
    """
    if len_generator is not None:
        pbar = tqdm(total=len_generator, unit='batch', desc='Extract')
    else:
        pbar = None
    features = []

    for data in data_generator:
        if pbar:
            pbar.update(1)
        feature = inception(data)[0].view(data.shape[0], -1)
        features.append(feature.to('cpu'))
    if pbar:
        pbar.close()
    features = torch.cat(features, 0)
    return features


@torch.no_grad()
def get_inception_features_mu_sigma(tensor, inception):
    features = inception(tensor)[0].view(tensor.shape[0], -1)
    features = features.cpu().numpy()

    sample_mu = np.mean(features, 0)
    sample_sigma = np.cov(features, rowvar=False).reshape((1, 1))
    return sample_mu, sample_sigma


def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.

    Args:
        mu1 (np.array): The sample mean over activations.
        sigma1 (np.array): The covariance matrix over activations for
            generated samples.
        mu2 (np.array): The sample mean over activations, precalculated on an
               representative data set.
        sigma2 (np.array): The covariance matrix over activations,
            precalculated on an representative data set.

    Returns:
        float: The Frechet Distance.
    """
    assert mu1.shape == mu2.shape, 'Two mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, ('Two covariances have different dimensions')

    cov_sqrt, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # Product might be almost singular
    if not np.isfinite(cov_sqrt).all():
        print('Product of cov matrices is singular. Adding {eps} to diagonal of cov estimates')
        offset = np.eye(sigma1.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))
            raise ValueError(f'Imaginary component {m}')
        cov_sqrt = cov_sqrt.real

    mean_diff = mu1 - mu2
    mean_norm = mean_diff @ mean_diff
    trace = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(cov_sqrt)
    fid = mean_norm + trace

    return fid


@torch.no_grad()
def compute_mmd(x, y):
    # 计算两个特征向量集合之间的最大均值差异（MMD）
    xx = torch.matmul(x, x.t())
    yy = torch.matmul(y, y.t())
    xy = torch.matmul(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    mmd = (torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy))
    return mmd


def calc_kid(real_features, fake_features):
    # 计算 KID
    real_features = F.normalize(real_features, p=2, dim=1)
    fake_features = F.normalize(fake_features, p=2, dim=1)
    mmd = compute_mmd(real_features, fake_features)
    return mmd.cpu().numpy()
