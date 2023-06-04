import os
import random
import numpy as np
import glob
from tqdm import tqdm
import math
import cv2
import functools
from einops import rearrange


from collections import OrderedDict

import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist

from pathlib import Path
import datetime

from omegaconf import OmegaConf

from trainer.logger import MessageLogger, AverageMeter
from trainer.utils import get_obj_from_str, calculate_parameters, tensor2img, calc_psnr_ssim, normalize
from measure.measurement import SuperResolutionOperator, ColorizationOperator, DeblurGaussOperator, SuperResolutionAverageOperator

from trainer.feature_inject import FeatureStore, register_feature_store, register_feature_store_celeba

import time

# from ldm.modules.diffusionmodules.util import make_ddim_timesteps

class BaseTrainer(object):
    def __init__(self, config):
        self.config = config

        self.init_device()
        self.init_seed()
        self.init_logger()

        if self.rank == 0:
            self.logger.print_log(OmegaConf.to_yaml(self.config))

        self.build_model()
        self.init_optimizaton()

        self.itr = 0
        if self.config.resume:
            self.resume_from_ckpt()

    def init_device(self):
        if self.config.gpu_id:
            print(torch.cuda.is_available())
            gpu_id = self.config.gpu_id
            num_gpu = len(gpu_id)
            # os.environ['CUDA_DEVICE_ORDER'] = 'PCU_BUS_ID'
            print(torch.cuda.is_available())
            # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([gpu_id[ii] for ii in range(num_gpu)])
            print(os.environ['CUDA_VISIBLE_DEVICES'])
            print(torch.cuda.is_available())
            self.num_gpu = num_gpu
        else:
            self.num_gpu = torch.cuda.device_count()

        if self.num_gpu > 1:
            # if torch.multiprocessing.get_start_method(allow_none=True) is None:
            #     torch.multiprocessing.set_start_method('spawn')
            dist.init_process_group('gloo')
            rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(rank % self.num_gpu)

        self.rank = int(os.environ['LOCAL_RANK']) if self.num_gpu > 1 else 0

    def init_seed(self):
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def init_logger(self, auto_change=True):
        name = self.config.exp_name
        save_dir = Path(self.config.save_dir) / name
        self.logger = None
        if self.rank == 0:
            if not self.config.resume:
                if save_dir.exists() and auto_change:
                    name = save_dir.name + '_' + str(datetime.datetime.now())[0:19].replace(' ', '_').replace(':', '_')
                    save_dir = save_dir.with_name(name)
                save_dir.mkdir()

                self.ckpt_dir = save_dir / 'ckpt'
                self.ckpt_dir.mkdir()

            tb_dir = Path(self.config.tb_dir) / name
            if not tb_dir.exists():
                tb_dir.mkdir()

            self.logger = MessageLogger(save_dir, tb_dir)

        self.loss_meter = AverageMeter()

    def build_model(self):
        params = self.config.model.get('param', dict)
        model = get_obj_from_str(self.config.model.model_name)(**params)
        print(self.rank)
        if self.num_gpu > 1:
            self.model = DDP(model.to(self.rank), device_ids=[self.rank, ])
        else:
            self.model = model.cuda()

        # print model info
        if self.rank == 0:
            num_params = calculate_parameters(self.model)
            self.logger.print_log("Model architecture:")
            self.logger.print_log(self.model.__repr__())
            self.logger.print_log("=======================================")
            self.logger.print_log("Number of parameters: {:.2f}M".format(num_params / 1e6))

    def init_optimizaton(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.config.train.lr,
                                     weight_decay=self.config.train.weight_decay)

        cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer = optimizer,
            T_max=self.config.train.total_itr,
            eta_min=0,
            last_epoch=-1
        )
        self.optimizer = optimizer
        self.scheduler = cosineScheduler

    def resume_from_ckpt(self):
        ckpt_path = self.config.resume

        if self.rank == 0:
            self.logger.print_log(f"Loading checkpoint from {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        self.model.load_state_dict(checkpoint['model'])
        torch.cuda.empty_cache()

        self.itr = checkpoint['itr']

        self.optimizer.load_state_dict(checkpoint['optim'])

    def save_ckpt(self):
        ckpt_path = self.ckpt_dir / "iter_{:d}.pkl".format(self.itr)
        if self.num_gpu > 1:
            torch.save({"model": self.model.module.state_dict(),
                        "optim": self.optimizer.state_dict(),
                        "iter": self.itr}, ckpt_path)
        else:
            torch.save({"model": self.model.state_dict(),
                        "optim": self.optimizer.state_dict(),
                        "iter": self.itr}, ckpt_path)

    def build_dataloader(self, type='train'):
        def _wrap_loader(loader):
            while True: yield from loader

        if type == 'train':
            params = self.config.datasets.train.get('params', dict)
            train_set = get_obj_from_str(self.config.datasets.train.target)(**params)

            if self.num_gpu > 1:
                shuffle = False
                sampler = DistributedSampler(train_set, num_replicas=self.num_gpu, rank=self.rank)
            else:
                shuffle = True
                sampler = None
            train_loader = DataLoader(dataset=train_set,
                                      batch_size=self.config.dataloader.train.batch_size // self.num_gpu,
                                      shuffle=shuffle,
                                      sampler=sampler,
                                      num_workers=self.config.dataloader.train.num_workers // self.num_gpu,
                                      pin_memory=True,
                                      drop_last=False,
                                      )
            self.train_loader = _wrap_loader(train_loader)

        elif type == 'test':
            params = self.config.datasets.test.get('params', dict)
            test_set = get_obj_from_str(self.config.datasets.test.target)(**params)

            if self.num_gpu > 1:
                shuffle = False
                sampler = DistributedSampler(test_set, num_replicas=self.num_gpu, rank=self.rank)
            else:
                shuffle = False
                sampler = None

            test_loader = DataLoader(dataset=test_set,
                                     batch_size=self.config.dataloader.test.batch_size // self.num_gpu,
                                     shuffle=shuffle,
                                     sampler=sampler,
                                     num_workers=self.config.dataloader.test.num_workers // self.num_gpu,
                                     pin_memory=True,
                                     drop_last=False,
                                     )
            self.test_loader = test_loader

    def train(self):
        train_sets = sorted(glob.glob(self.config.data.train.data_path))

        self.model.train()
        pbar = tqdm(total=self.config.train.total_itr, unit='chunk')

        itr = self.itr
        while itr < self.config.train.total_itr:
            for train_set_path in train_sets:
                train_loader = self.build_dataloader(train_set_path)

                for k, data in enumerate(train_loader):
                    if itr > self.config.train.total_itr:
                        break

                    self.train_step(data)

                    itr += 1
                    self.itr = itr
                    pbar.update(1)

                    if itr % self.config.train.val_freq == 0:
                        if self.rank == 0:
                            self.valid()

                    if itr % self.config.train.save_freq == 0:
                        self.save_ckpt()

                    if itr % self.config.train.log_freq == 0:
                        pass

    def train_step(self, data):
        pass

    def valid(self):
        pass

class DiffIRInference(BaseTrainer):
    def __init__(self, config):
        self.config = config

        self.init_device()
        self.init_seed()
        self.init_logger()

        # self.build_model()
        self.build_sampler()

    def init_logger(self):
        name = self.config.exp_name
        save_dir = Path(self.config.save_dir) / name
        if self.rank == 0:

            self.save_dir = save_dir
            self.process_img_path = save_dir / "process"
            self.restored_img_path = save_dir / "restored"
            self.lq_img_path = save_dir / "lq"

            if not self.save_dir.exists():
                self.save_dir.mkdir()

            if not self.process_img_path.exists():
                self.process_img_path.mkdir()

            if not self.restored_img_path.exists():
                self.restored_img_path.mkdir()

            if not self.lq_img_path.exists():
                self.lq_img_path.mkdir()

    @staticmethod
    def _load_model(model, ckpt):
        if list(model.state_dict().keys())[0].startswith('module.'):
            if list(ckpt.keys())[0].startswith('module.'):
                ckpt = ckpt
            else:
                ckpt = OrderedDict({f'module.{key}': value for key, value in ckpt.items()})
        else:
            if list(ckpt.keys())[0].startswith('module.'):
                ckpt = OrderedDict({key[7:]: value for key, value in ckpt.items()})
            else:
                ckpt = ckpt
        model.load_state_dict(ckpt, strict=True)
        model.cuda()

    def build_sampler(self):
        from trainer.sampler import DDIMSampler

        diffusion_cfg = OmegaConf.load(self.config.diffusion.cfg_path).diffusion
        diffusion_cfg.params.forward_timestep = self.config.forward_timestep
        diffusion_cfg.params.backward_timestep = self.config.backward_timestep
        diffusion_cfg.params.insert_step = self.config.insert_step
        diffusion_cfg.params.align_place = self.config.aligned_timestep


        diffusion_for, diffusion_back = get_obj_from_str(diffusion_cfg.target)(**(diffusion_cfg.params))

        diffusion_for_timestep = diffusion_for.timestep_map
        diffusion_back_timestep = diffusion_back.timestep_map
        self.aligned_timestep = [diffusion_for_timestep[i] for i in self.config.aligned_timestep]
        print("Forward timestep: ", diffusion_for_timestep)
        print("Backward timestep: ", diffusion_back_timestep)
        print("Aligned timestep: ", self.aligned_timestep)

        diffusion_model_cfg = OmegaConf.load(self.config.diffusion.cfg_path).model
        diffusion_model = get_obj_from_str(diffusion_model_cfg.target)(**(diffusion_model_cfg.params))
        diffusion_model_ckpt = torch.load(self.config.diffusion.resume_path)
        self._load_model(diffusion_model, diffusion_model_ckpt)
        # diffusion_model.convert_to_fp16()
        diffusion_model.eval()

        if hasattr(self.config, 'classifier'):
            classifier_model_cfg = OmegaConf.load(self.config.classifier.cfg_path).model
            classifier_model = get_obj_from_str(classifier_model_cfg.target)(**(classifier_model_cfg.params))
            classifier_model_ckpt = torch.load(self.config.classifier.resume_path)
            self._load_model(classifier_model, classifier_model_ckpt)
            self.classifier_model = classifier_model
            self.classifier_model.eval()
        else:
            self.classifier_model = None

        self.sampler = DDIMSampler(diffusion_for=diffusion_for,
                                   diffusion_back=diffusion_back,
                                   model=diffusion_model,
                                   img_size=self.config.sample.img_size,
                                   noise_eta=self.config.noise_eta,
                                   save_process=self.config.save_process,)

        self.forward_timestep = self.config.forward_timestep
        self.backward_timestep = self.config.backward_timestep
        self.noise_eta = self.config.noise_eta
        self.use_consistency = self.config.use_consistency

        self.use_ddim = self.config.sample.use_ddim
        self.eta = self.config.sample.ddim_eta

        self.save_process = self.config.save_process

    def image_restore_naive(self, lq):
        b = lq.shape[0]

        # inverse
        x_0 = self.deg_operator.transpose(lq)
        x_t = self.sampler.q_sample(x_start=x_0, t=torch.tensor([self.forward_timestep - 1] * b).cuda())

        back_xt_list = []
        back_x0_list = []

        # generate
        indices = list(range(self.backward_timestep))[::-1]

        for i in indices:
            t = torch.tensor([i] * b).cuda()
            out = self.sampler.p_sample(x=x_t, t=t, use_ddim=self.use_ddim)
            x_t = out['sample']

            if i in self.sampler.save_index_back:
                back_xt_list.append(normalize(x_t, reverse=True))
                back_x0_list.append(normalize(out['pred_xstart'], reverse=True))

        if self.save_process:
            back_xt = torch.cat(back_xt_list[::-1], dim=3)
            back_x0 = torch.cat(back_x0_list[::-1], dim=3)
            out_process = torch.cat([back_xt, back_x0], dim=2)
        else:
            out_process = None

        return normalize(x_t, reverse=True), out_process

    def image_restore_ddnm(self, lq):
        b = lq.shape[0]

        n = self.forward_timestep
        t = torch.tensor([n - 1] * b).cuda()

        # inverse
        x_0 = self.deg_operator.transpose(lq)
        x_t = self.sampler.q_sample(x_start=x_0, t=torch.tensor([self.forward_timestep - 1] * b).cuda())
        back_xt_list = []
        back_x0_list = []

        resample_n = 1
        # generate
        n = self.backward_timestep
        indices = list(range(n))[::-1]

        for i in indices:
            use_consistency = True
            t = torch.tensor([i] * b).cuda()

            for j in range(resample_n):
                x_t, x_0_pred = self.sampler.p_sample_svd(img_guidance=lq, x=x_t, t=t,
                                                   eta=self.eta, use_consistency=use_consistency,
                                                   operator=self.deg_operator, use_ddim=self.use_ddim)
                if j != resample_n - 1:
                    x_t = self.sampler.resample(x_t, i, i + 1)

            if i in self.sampler.save_index_back:
                back_xt_list.append(normalize(x_t, reverse=True))
                back_x0_list.append(normalize(x_0_pred, reverse=True))

        if self.save_process:
            back_xt = torch.cat(back_xt_list[::-1], dim=3)
            back_x0 = torch.cat(back_x0_list[::-1], dim=3)
            out_process = torch.cat([back_xt, back_x0], dim=2)
        else:
            out_process = None

        return normalize(x_t, reverse=True), out_process

    def image_restore_inversion(self, lq):
        b = lq.shape[0]

        n = self.forward_timestep
        t = torch.tensor([n] * b).cuda()

        # inverse
        indices = list(range(n - 1))

        x_0 = self.deg_operator.transpose(lq)
        x_t = x_0
        for_xt_list = []
        for_x0_list = []

        # todo: add resample strategy for add noise process(2023.04.06)
        # notice that resample will need triple time cost
        # and the story is unconvincing
        # todo or not to do, that's a question

        for i in indices:
            t = torch.tensor([i] * b).cuda()
            out = self.sampler.q_sample_ddim(x=x_t, t=t, add_noise=True, clip_denoised=True,
                                             use_consistency=False, operator=self.deg_operator,
                                             img_guidance=lq)
            x_t = out['sample']
            if i in self.sampler.save_index_for:
                for_xt_list.append(normalize(x_t, reverse=True))
                for_x0_list.append(normalize(out['pred_xstart'], reverse=True))

        # todo: add resample strategy for denoise process(2023.04.06)
        # strategy1: t+1 -> t -> t+1 -> t , repeat n times (Repaint)
        # strategy2: t+1 -> t -> t+j -> t+j-1 -> ... -> t, resample p(t+j|t) (DDNM)
        resample_n = 1

        # generate
        n = self.backward_timestep
        indices = list(range(n))[::-1]
        back_xt_list = []
        back_x0_list = []

        for i in indices:
            use_consistency = False
            t = torch.tensor([i] * b).cuda()

            for j in range(resample_n):
                x_t, x_0_pred = self.sampler.p_sample_svd(img_guidance=lq, x=x_t, t=t,
                                                          eta=self.eta, use_consistency=use_consistency,
                                                          operator=self.deg_operator, use_ddim=self.use_ddim)
                if j != resample_n - 1:
                    x_t = self.sampler.resample(x_t, i, i + 1)
            if i in self.sampler.save_index_back:
                back_xt_list.append(normalize(x_t, reverse=True))
                back_x0_list.append(normalize(x_0_pred, reverse=True))

        if self.save_process:
            for_xt = torch.cat(for_xt_list, dim=3)
            for_x0 = torch.cat(for_x0_list, dim=3)
            back_xt = torch.cat(back_xt_list[::-1], dim=3)
            back_x0 = torch.cat(back_x0_list[::-1], dim=3)
            out_process = torch.cat([for_xt, for_x0, back_xt, back_x0], dim=2)
        else:
            out_process = None

        return normalize(x_t, reverse=True), out_process

    def image_restore_inject(self, lq):
        # build feature storer
        fea_storer = FeatureStore(self.config.sample.inject_feature)
        if 'uncond' in self.config.diffusion.cfg_path:
            register_feature_store(self.sampler.model, fea_storer)
        elif 'celeba' in self.config.diffusion.cfg_path:
            register_feature_store_celeba(self.sampler.model, fea_storer)

        b = lq.shape[0]

        # inverse
        n = self.forward_timestep
        indices = list(range(n - 1))

        x_0 = self.deg_operator.transpose(lq)
        x_t = x_0
        for_xt_list = []
        for_x0_list = []

        for i in indices:
            t_ori = self.sampler.get_ori_timesteps(i, is_forward=True)
            if t_ori in self.aligned_timestep:
                fea_storer.set_state('store')
            else:
                fea_storer.set_state('nothing')

            t = torch.tensor([i] * b).cuda()
            out = self.sampler.q_sample_ddim(x=x_t, t=t, add_noise=True, clip_denoised=True,
                                             use_consistency=False, operator=self.deg_operator,
                                             img_guidance=lq)
            x_t = out['sample']

            if i in self.sampler.save_index_for:
                for_xt_list.append(normalize(x_t, reverse=True))
                for_x0_list.append(normalize(out['pred_xstart'], reverse=True))

        # generate
        n = self.backward_timestep
        indices = list(range(n))[::-1]
        back_xt_list = []
        back_x0_list = []

        for i in indices:
            use_consistency = self.use_consistency
            t = torch.tensor([i] * b).cuda()

            t_ori = self.sampler.get_ori_timesteps(i, is_forward=False)
            if t_ori in self.aligned_timestep:
                fea_storer.set_state('inject')
            else:
                fea_storer.set_state('nothing')

            x_t, x_0_pred = self.sampler.p_sample_svd(img_guidance=lq, x=x_t, t=t,
                                                      eta=self.eta, use_consistency=use_consistency,
                                                      operator=self.deg_operator, use_ddim=self.use_ddim)

            if i in self.sampler.save_index_back:
                back_xt_list.append(normalize(x_t, reverse=True))
                back_x0_list.append(normalize(x_0_pred, reverse=True))

        if self.save_process:
            for_xt = torch.cat(for_xt_list, dim=3)
            for_x0 = torch.cat(for_x0_list, dim=3)
            back_xt = torch.cat(back_xt_list[::-1], dim=3)
            back_x0 = torch.cat(back_x0_list[::-1], dim=3)
            out_process = torch.cat([for_xt, for_x0, back_xt, back_x0], dim=2)
        else:
            out_process = None

        return normalize(x_t, reverse=True), out_process

    def image_restore(self, lq):
        if self.config.method == 'naive':
            return self.image_restore_naive(lq)
        elif self.config.method == 'ddnm':
            return self.image_restore_ddnm(lq)
        elif self.config.method == 'inversion_only':
            return self.image_restore_inversion(lq)
        elif self.config.method == 'inject':
            return self.image_restore_inject(lq)
        else:
            raise ValueError(f"The argument Method is not in range. Expected [naive, ddnm, inversion_only, inject], Given {self.config.method}")

    def eval(self):
        print("begin")
        self.build_dataloader(type='test')
        print(len(self.test_loader))

        device = torch.tensor(0.0).cuda().device

        # todo: Add other degradation
        if self.config.degradation == 'sr_bicubic':
            self.deg_operator = SuperResolutionOperator(scale=self.config.factor)
        elif self.config.degradation == 'sr_average':
            self.deg_operator = SuperResolutionAverageOperator(scale=self.config.factor)
        elif self.config.degradation == 'colorization':
            self.deg_operator = ColorizationOperator()
        elif self.config.degradation == 'deblur_gauss':
            self.deg_operator = DeblurGaussOperator()

        for k, data in enumerate(self.test_loader):
            data = self.prepare_data(data)
            hq = data['gt']
            hq = normalize(hq)

            lq = self.deg_operator.forward(hq)
            name = data['name']
            hq_pred, out_process = self.image_restore(lq)

            for i in range(hq.shape[0]):
                cv2.imwrite(str(self.lq_img_path / "{}.png".format(name[i])), tensor2img(normalize(lq[i], reverse=True)))
                cv2.imwrite(str(self.restored_img_path / "{}.png".format(name[i])), tensor2img(hq_pred[i]))
                if self.save_process:
                    cv2.imwrite(str(self.process_img_path / "{}_process.jpg".format(name[i])), tensor2img(out_process[i]))
                print(name[i])

            # prevent potential memory leak
            if k % 10 == 0:
                torch.cuda.empty_cache()

    @torch.no_grad()
    def prepare_data(self, data):
        return {key: value.cuda() if torch.is_tensor(value) else value for key, value in data.items()}












