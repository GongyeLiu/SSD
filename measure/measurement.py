import torch.nn.functional as F
from functools import partial
from measure.resize_right import resize
from measure import interp_methods

from functions.svd_operators import SRConv, Colorization, Deblurring, WalshAadamardCS

import numpy as np
import torch

class LinearOperator(object):
    def forward(self, x):
        # calcualte H*x
        pass

    def transpose(self, x):
        # calculate H^-1 * x
        pass

    def project(self, x, y):
        # calculate (I - H^-1 * H) X + H^-1 y
        out = x - self.transpose(self.forward(x)) + self.transpose(y)
        return out


# class SuperResolutionOperator(LinearOperator):
#     def __init__(self, scale=4.0):
#         self.scale = scale
#
#     def forward(self, x):
#         # downsample
#         out = resize(x, scale_factors=1. / self.scale)
#         return out
#
#
#     def transpose(self, x):
#         # upsample
#         out = resize(x, scale_factors=self.scale)
#         return out


class SuperResolutionOperator(LinearOperator):
    def __init__(self, scale=4.0, channels=3, img_size=256):
        scale = int(scale)
        self.scale = scale

        def bicubic_kernel(x, a=-0.5):
            if abs(x) <= 1:
                return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
            elif 1 < abs(x) and abs(x) < 2:
                return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
            else:
                return 0

        k = np.zeros((scale * 4))
        for i in range(scale * 4):
            x = (1 / scale) * (i - np.floor(scale * 4 / 2) + 0.5)
            k[i] = bicubic_kernel(x)
        k = k / np.sum(k)
        kernel = torch.from_numpy(k).float().cuda()
        A_funcs = SRConv(kernel / kernel.sum(), channels, img_size, kernel.device, stride=scale)
        self.A_funcs = A_funcs

        self.img_size = img_size
        self.channels = channels
        self.img_size_y = img_size // scale

    def forward(self, x):
        # downsample
        out = self.A_funcs.A(x).view(x.shape[0], self.channels, self.img_size_y, self.img_size_y)
        return out


    def transpose(self, x):
        # upsample
        out = self.A_funcs.A_pinv(x).view(x.shape[0], self.channels, self.img_size, self.img_size)
        return out


def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out


class SuperResolutionAverageOperator(LinearOperator):
    def __init__(self, scale=4.0, channels=3, img_size=256):
        scale = int(scale)
        self.scale = scale

        A_funcs = torch.nn.AdaptiveAvgPool2d((256 // scale, 256 // scale))
        Apy_funcs = lambda z: MeanUpsample(z, scale)
        self.A_funcs = A_funcs
        self.Ap_funcs = Apy_funcs

        self.img_size = img_size
        self.channels = channels
        self.img_size_y = img_size // scale

    def forward(self, x):
        # downsample
        out = self.A_funcs(x)
        return out


    def transpose(self, x):
        # upsample
        out = self.Ap_funcs(x)
        return out


class ColorizationOperator(LinearOperator):
    def __init__(self, channels=3, img_size=256):

        self.A_funcs = Colorization(img_size)

        self.img_size = img_size
        self.channels = channels

    def forward(self, x):
        out = self.A_funcs.A(x).view(x.shape[0], 1, self.img_size, self.img_size)
        return out


    def transpose(self, x):
        out = self.A_funcs.A_pinv(x).view(x.shape[0], self.channels, self.img_size, self.img_size)
        return out


class DeblurGaussOperator(LinearOperator):
    def __init__(self, kernel_size=9, sigma=15, channels=3, img_size=256):
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        kernel = torch.Tensor([pdf(x) for x in range(-(kernel_size // 2), kernel_size // 2 + 1)]).cuda()
        A_funcs = Deblurring(kernel / kernel.sum(), channels, img_size)

        self.A_funcs = A_funcs
        self.img_size = img_size
        self.channels = channels

    def forward(self, x):
        out = self.A_funcs.A(x).view(x.shape[0], self.channels, self.img_size, self.img_size)
        return out

    def transpose(self, x):
        out = self.A_funcs.A_pinv(x).view(x.shape[0], self.channels, self.img_size, self.img_size)
        return out
    

class MaskedOperator(LinearOperator):
    def __init__(self, type='random_point', mask_size=[128, 128], mask_prob=0.5, img_size=256):
        self.type = type
        self.mask_size = mask_size
        self.mask_prob = mask_prob
        self.img_size = img_size
        self.mask_img = None

    def get_mask(self):
        if self.type == 'random_point':
            mask = torch.rand(1, 1, self.img_size, self.img_size) > self.mask_prob
            mask = mask.float().cuda()
        elif self.type == 'random_block':
            mask = torch.ones(1, 1, self.img_size, self.img_size)
            lt_point = torch.randint(0, self.img_size - self.mask_size[0], (1, 2))
            mask[:, :, lt_point[0, 0]:lt_point[0, 0] + self.mask_size[0], lt_point[0, 1]:lt_point[0, 1] + self.mask_size[1]] = 0
        elif self.type == "specific":
            mask = self.mask
        return mask
    
    def register_mask(self, mask):
        self.mask = mask

    def forward(self, x, mask=None):
        if mask is None:
            mask = self.get_mask()

        return x * mask        

    def transpose(self, x, mask=None):
        if mask is None:
            mask = self.get_mask()
        return x * mask
    

class WalshAadamardCSOperator(LinearOperator):
    def __init__(self, channels=3, img_size=256, ratio=0.25):
        with torch.random.fork_rng():
            torch.manual_seed(42)
            random_permutation = torch.randperm(img_size ** 2, device='cuda')
        A_funcs = WalshAadamardCS(channels, img_size, round(1 / ratio), random_permutation)
        self.A_funcs = A_funcs
        self.img_size = img_size
        self.channels = channels

    def forward(self, x):
        out = self.A_funcs.A(x)
        return out
    
    def transpose(self, x):
        out = self.A_funcs.A_pinv(x).view(x.shape[0], self.channels, self.img_size, self.img_size)
        return out