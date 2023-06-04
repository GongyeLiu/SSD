import logging
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from trainer.utils import tensor2img
import cv2
# import wandb
from collections import OrderedDict


def get_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    info_file_handler = logging.FileHandler(log_path, mode="a")
    logger.addHandler(info_file_handler)
    screen_file_handler = logging.StreamHandler()
    logger.addHandler(screen_file_handler)
    return logger


def init_tb_logger(log_dir):
    if log_dir == None:
        return None
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger


# def init_wandb_logger(opt):
#     if opt['logger']['wandb']['project'] is None:
#         return None
#     if opt['logger']['wandb']['resume_id'] is None:
#         id = wandb.util.generate_id()
#         opt['logger']['wandb']['resume_id'] = id
#         resume = "never"
#     else:
#         id = opt['logger']['wandb']['resume_id']
#         resume = "allow"
#     wandb.init(project=opt['logger']['wandb']['project'], id=id, resume=resume, config=opt, name=opt['name'])
#     return id

class Metric(object):
    def __init__(self, name, value):
        self.name = name
        self.total_value = value
        self.num = 1

    def update(self, value):
        self.total_value += value
        self.num += 1

    def is_empty(self):
        return self.num == 0

    def get_average(self, is_dict=False, is_reset=True):
        avg_value = self.total_value / self.num
        if is_dict:
            out = {self.name: avg_value}
        else:
            out = avg_value
        if is_reset:
            self.reset()
        return out

    def reset(self):
        self.total_value = 0
        self.num = 0


class AverageMeter(object):
    def __init__(self):
        self.metrics = OrderedDict()

    def update(self, metric_dict):
        for key, value in metric_dict.items():
            if key in self.metrics.keys():
                self.metrics[key].update(value)
            else:
                self.metrics[key] = Metric(key, value)

    def get_average(self, name_list=None, is_reset=True):
        out_dict = OrderedDict()
        if name_list is None:
            for key, value in self.metrics.items():
                if not value.is_empty():
                    out_dict[key] = value.get_average(is_reset=True)
        else:
            for name in name_list:
                if name in self.metrics.keys():
                    if not self.metrics[name].is_empty():
                        out_dict[name] = self.metrics[name].get_average(is_reset=True)

        return out_dict


class MessageLogger(object):
    def __init__(self, save_dir, logger_dir):
        log_path = save_dir / 'logs'
        tb_log_path = logger_dir
        self.logger = get_logger(log_path)
        self.tb_logger = init_tb_logger(tb_log_path)
        self.img_path = save_dir / 'images'

        if not self.img_path.exists():
            self.img_path.mkdir()

        # id = init_wandb_logger(opt)
        # if id is not None:
        #     self.print_log(id)
        # self.use_wandb = False if opt['logger']['wandb']['project'] is None else True
        # self.interval = opt['logger']['print_freq']

    def print_log(self, msg):
        self.logger.info(msg)

    def write(self, log_dict, itr):
        for key, value in log_dict.items():
            self.tb_logger.add_scalar(key, value, itr)
        self.tb_logger.flush()

    def save_images(self, images, itr, use_tb=True):
        if use_tb:
            self.tb_logger.add_images("Valid Images", images, itr, dataformats='HWC')
        img_dir = self.img_path / str(itr)
        if not img_dir.exists():
            img_dir.mkdir()
        cv2.imwrite((img_dir / "valid.jpg").name, images)






