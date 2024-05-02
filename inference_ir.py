import argparse
import os
from omegaconf import OmegaConf

# debug only
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from trainer.trainer import DiffIRInference


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--exp_name", type=str, default="demo_test",
                        help="Folder to save the checkpoints and training log")
    parser.add_argument("--save_dir", type=str, default="./data_demo/restored_results/",
                                             help="Folder to save the checkpoints and training log")
    parser.add_argument("--cfg_path", type=str, default="./configs/inference/celeba_256.yaml",
                                                                        help="Configs of yaml file")
    parser.add_argument("--degradation", type=str, default='sr_bicubic',
                        help='degradation type, available: [colorization, deblur_gauss, sr_bicubic, sr_average]')
    parser.add_argument("--deg_noise", type=str, default='none',
                        help='noise type, available: [none, jpeg, gauss]')
    parser.add_argument("--def_factor", type=float, default=4.)
    parser.add_argument("--forward_timestep", type=int, default=15)
    parser.add_argument("--backward_timestep", type=int, default=85)
    parser.add_argument("--insert_step", type=int, default=550)
    parser.add_argument("--stop_consistency_timestep", type=int, default=200)
    parser.add_argument("--noise_eta", type=float, default=0.4)
    parser.add_argument("--method", type=str, default="ssd")
    parser.add_argument("--save_process", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--mask_type", type=str, default='specific',
                        help='mask type, available: [specific, random_block, random_point]')
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_parser()

    configs = OmegaConf.load(args.cfg_path)
    for key in vars(args):
        configs[key] = getattr(args, key)

    inferencer = DiffIRInference(configs)
    inferencer.eval()




