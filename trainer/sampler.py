import numpy as np
import torch
import torch.nn.functional as F
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from arcFace.loss import IdentityLoss
from guided_diffusion.gaussian_diffusion import _extract_into_tensor
from random import randrange
from trainer.utils import tensor2img, normalize
import cv2
from scipy.stats import norm
import numpy as np


class DDIMSampler(object):
    def __init__(self, diffusion_for, diffusion_back, model, img_size, noise_eta=0.0, save_process=False):
        self.diffusion_for = diffusion_for
        self.diffusion_back = diffusion_back
        self.model = model
        self.img_size = img_size

        self.noise_eta = noise_eta
        self._get_forward_coeffs(noise_eta)

        if save_process:
            self.save_process = True
            n_save = min(7, self.diffusion_for.num_timesteps - 1, self.diffusion_back.num_timesteps - 1)
            self.save_index_for = list(np.linspace(0, self.diffusion_for.num_timesteps - 2, n_save, dtype=np.int))
            self.save_index_back = list(np.linspace(0, self.diffusion_back.num_timesteps - 2, n_save, dtype=np.int))
        else:
            self.save_process = False
            self.save_index_for = []
            self.save_index_back = []

    def _get_forward_coeffs(self, noise_eta):
        betas = self.diffusion_for.betas
        betas_next = np.append(betas[1:], 0.0)
        alphas_cumprod_next = self.diffusion_for.alphas_cumprod_next
        self.coeff_1 = np.sqrt(1 - alphas_cumprod_next - noise_eta * betas_next)
        self.coeff_2 = np.sqrt(noise_eta * betas_next)

    def calc_mse_loss(self, x, y):
        mse = torch.linalg.norm(x - y)
        # print(mse.item())
        # loss = mse / mse.detach()
        loss = mse
        return loss

    def auto_corr_loss(self, x, random_shift=True):
        B, C, H, W = x.shape
        assert B == 1
        x = x.squeeze(0)
        # x must be shape [C,H,W] now
        reg_loss = 0.0
        for ch_idx in range(x.shape[0]):
            noise = x[ch_idx][None, None, :, :]
            while True:
                if random_shift:
                    roll_amount = randrange(noise.shape[2] // 2)
                else:
                    roll_amount = 1
                reg_loss += (noise * torch.roll(noise, shifts=roll_amount, dims=2)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=roll_amount, dims=3)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        return reg_loss

    def kl_loss(self, x):
        mu = x.mean()
        var = x.var()
        return 0.5 * (var + mu ** 2 - 1 - torch.log(var + 1e-5))

    def kl_divergence(self, x):

        p = norm.pdf(x)

        q = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

        kl = np.sum(np.where(p != 0, p * np.log(p / q), 0))

        return kl

    @torch.no_grad()
    def sample_ddim(
            self,
            batch_size,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
    ):

        out = self.diffusion_for.ddim_sample_loop(
            model=self.model,
            shape=(batch_size, 3, self.img_size, self.img_size),
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        )

        return out

    def q_sample_guidance(self, x, t, img_guidance, s=1.0):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            out = self.diffusion_back.q_sample_step(model=self.model, x=x, t=t, clip_denoised=True)
            loss = torch.tensor(0.0).cuda()
            mse_loss = self.calc_mse_loss(out['pred_xstart'], img_guidance)
            # identity_loss = self.id_loss(gt=img_guidance, pred=out['pred_xstart'])
            # loss = identity_loss + 5 * mse_loss
            loss = mse_loss
            grad = -s * torch.autograd.grad(loss, x)[0]

        with torch.no_grad():
            sample = out['sample'].float() + out['variance'] * grad.float()
            out['sample'] = sample
            return out


    @torch.no_grad()
    def p_sample(
            self,
            x,
            t,
            use_ddim=False,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        if use_ddim:
            out = self.p_sample_ddim(x=x, t=t, clip_denoised=clip_denoised, eta=eta)
            # out = self.diffusion.ddim_sample(
            #     model=self.model,
            #     x=x,
            #     t=t,
            #     clip_denoised=clip_denoised,
            #     denoised_fn=denoised_fn,
            #     cond_fn=cond_fn,
            #     model_kwargs=model_kwargs,
            #     eta=0.0
            # )

        else:
            out = self.diffusion_back.p_sample(
                model=self.model,
                x=x,
                t=t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
            )
        return out

    def p_sample_ddim(
            self,
            x,
            t,
            clip_denoised=True,
            eta=0.0,
    ):
        eps = self.diffusion_back.get_model_out(model=self.model, x=x, t=t)
        x_0_pred = self.diffusion_back._predict_xstart_from_eps(x_t=x, t=t, eps=eps)

        if clip_denoised:
            x_0_pred = x_0_pred.clamp(-1, 1)

        alpha_bar = _extract_into_tensor(self.diffusion_back.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.diffusion_back.alphas_cumprod_prev, t, x.shape)
        sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise = torch.randn_like(x)
        mean_pred = (
                x_0_pred * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": x_0_pred, 'noise': eps}

    def p_sample_guidance(self, img_guidance, x, t, s=5.0, use_ddim=False):

        with torch.enable_grad():
            x = x.detach().requires_grad_()
            out = self.diffusion_back.p_mean_variance(self.model, x, t)

            loss = torch.tensor(0.0).cuda()

            mse_loss = self.calc_mse_loss(out['pred_xstart'], img_guidance)
            loss = mse_loss
            grad = -s * torch.autograd.grad(loss, x)[0]

        with torch.no_grad():
            if use_ddim:
                eps = self.diffusion_back._predict_eps_from_xstart(x, t, out["pred_xstart"])
                alpha_bar_prev = _extract_into_tensor(self.diffusion_back.alphas_cumprod_prev, t, x.shape)
                mean = (
                        out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
                        + torch.sqrt(1 - alpha_bar_prev) * eps
                )
                sample = mean.float() + grad.float()

            else:
                noise = torch.randn_like(x)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                )

                mean = out['mean'].float() + grad.float()

                sample = mean + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

        return sample.detach()

    @torch.no_grad()
    def p_sample_svd(self, img_guidance, x, t, operator, eta=0.0, use_ddim=False, use_consistency=True):
        # calc score function
        eps = self.diffusion_back.get_model_out(model=self.model, x=x, t=t)
        x_0_pred = self.diffusion_back._predict_xstart_from_eps(x_t=x, t=t, eps=eps)
        x_0_pred = x_0_pred.clamp(-1, 1)

        # img_guidance_resize = operator.forward(img_guidance)
        if use_consistency:
            x_0_pred = operator.project(x=x_0_pred, y=img_guidance)

        if use_ddim:
            alpha_bar_prev = _extract_into_tensor(self.diffusion_back.alphas_cumprod_prev, t, x.shape)
            c1 = (1 - eta ** 2) ** 0.5 * torch.sqrt(1 - alpha_bar_prev)
            c2 = eta * torch.sqrt(1 - alpha_bar_prev)
            mean = (
                    x_0_pred * torch.sqrt(alpha_bar_prev)
                    + c1 * eps
                    + c2 * torch.randn_like(x_0_pred)
            )
            sample = mean.float()

        else:
            alpha_bar_prev = _extract_into_tensor(self.diffusion_back.alphas_cumprod_prev, t, x.shape)
            alpha_bar = _extract_into_tensor(self.diffusion_back.alphas_cumprod, t, x.shape)
            sigma = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) \
                    * torch.sqrt(1 - (alpha_bar / alpha_bar_prev))

            c1 = torch.sqrt(1 - alpha_bar_prev - sigma ** 2)
            c2 = sigma
            mean = (
                    x_0_pred * torch.sqrt(alpha_bar_prev)
                    + c1 * eps
                    + c2 * torch.randn_like(x_0_pred)
            )
            sample = mean.float()

        return sample.detach(), x_0_pred.detach()


    @torch.no_grad()
    def q_sample(self, x_start, t, noise=None):
        return self.diffusion_for.q_sample(x_start, t, noise)

    @torch.no_grad()
    def resample(self, x_t, i, j):
        # resample p(x_j | x_i)
        # added in 2023.04.06, unchecked
        b = x_t.shape[0]
        while i < j:
            t = torch.tensor([i] * b).cuda()
            beta_t_prev = _extract_into_tensor(self.diffusion_back.betas_prev, t, x_t.shape)
            x_t = torch.sqrt(1 - beta_t_prev) * x_t + torch.sqrt(beta_t_prev) * torch.randn_like(x_t)
            i += 1
        return x_t

    def q_sample_ddim(self, x, t, clip_denoised=True, add_noise=False, noise_eta=None, use_consistency=False, operator=None, img_guidance=None):
        with torch.no_grad():
            eps = self.diffusion_for.get_model_out(model=self.model, x=x, t=t)

        # print(torch.max(eps))
        # print(torch.min(eps))

        # eps = eps.clamp(-3, 3)

        a = eps.mean()
        b = eps.std()
        # print("t:{:d} | mean:{:.4f} | std:{:.4f}".format(t.item(), a.item(), b.item()))

        x_0_pred = self.diffusion_for._predict_xstart_from_eps(x_t=x, t=t, eps=eps)

        if use_consistency:
            x_0_pred = operator.project(x=x_0_pred, y=img_guidance)

        if clip_denoised:
            x_0_pred = x_0_pred.clamp(-1, 1)

        alpha_bar_next = _extract_into_tensor(self.diffusion_for.alphas_cumprod_next, t, x.shape)
        beta_t_next = _extract_into_tensor(self.diffusion_for.betas, t + 1, x.shape)

        if noise_eta is not None:
            c1 = torch.sqrt(1 - alpha_bar_next - noise_eta * beta_t_next)
            c2 = torch.sqrt(noise_eta * beta_t_next)
            # print(noise_eta)
        else:
            c1 = _extract_into_tensor(self.coeff_1, t, x.shape)
            c2 = _extract_into_tensor(self.coeff_2, t, x.shape)
        # print(c1[0, 0, 0, 0].item(), c2[0, 0, 0, 0].item())

        if add_noise:
            # mean_pred = torch.sqrt(1 - beta_t_next) * x + torch.sqrt(beta_t_next) * torch.randn_like(x)
            mean_pred = (
                    x_0_pred * torch.sqrt(alpha_bar_next)
                    + c1 * eps
                    + c2 * torch.randn_like(x_0_pred)
            )
        else:
            mean_pred = (
                    x_0_pred * torch.sqrt(alpha_bar_next)
                    + torch.sqrt(1 - alpha_bar_next) * eps
            )

        sample = mean_pred
        return {"sample": sample, "pred_xstart": x_0_pred, 'noise': eps}
        # return sample

    def get_ori_timesteps(self, t, is_forward=True):
        if is_forward:
            return self.diffusion_for.timestep_map[t]
        else:
            return self.diffusion_back.timestep_map[t]









