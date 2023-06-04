from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import random
import h5py
import cv2
import glob
from PIL import Image


def imread(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255
    img = img.transpose((2, 0, 1))
    return img


def imread_PIL(img_path):
    with open(img_path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class BasePairedDataset(Dataset):
    def __init__(self,
                 lq_root=None,
                 gt_root=None
                 ):
        self.lq_path = sorted(glob.glob(lq_root))
        self.gt_path = sorted(glob.glob(gt_root))
        assert len(self.lq_path) == len(self.gt_path)

    def __getitem__(self, item):
        lq = imread(self.lq_path[item])
        gt = imread(self.gt_path[item])
        name = self.lq_path[item].split('/')[-1].split('.')[0]
        return {'lq': lq, 'gt': gt, 'name': name}

    def __len__(self):
        return len(self.lq_path)


class BaseSingleDataset(Dataset):
    def __init__(self,
                 lq_root=None,
                 ):
        self.lq_path = sorted(glob.glob(lq_root))

    def __getitem__(self, item):
        name = self.lq_path[item].split('/')[-1].split('.')[0]
        lq = imread(self.lq_path[item])
        return {'lq': lq, 'name': name}

    def __len__(self):
        return len(self.lq_path)


class CenterCropLongEdge(object):
    """Crops the given PIL Image on the long edge.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


class ImageDataset(Dataset):
    def __init__(self, gt_root, img_size=128):
        self.gt_path = sorted(glob.glob(gt_root))
        self.transform = transforms.Compose([
            CenterCropLongEdge(),
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

    def __getitem__(self, item):
        name = self.gt_path[item].split('/')[-1].split('.')[0]
        gt = imread_PIL(self.gt_path[item])
        gt = self.transform(gt)
        return {'gt': gt, 'name': name}

    def __len__(self):
        return len(self.gt_path)

