import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps

NUM_CLASSES = 1000


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def classifier_defaults():
    """
    Defaults for classifier models.
    """
    return dict(
        image_size=64,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_resblock_updown=True,  # False
        classifier_pool="attention",
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )
    res.update(diffusion_defaults())
    return res


def classifier_and_diffusion_defaults():
    res = classifier_defaults()
    res.update(diffusion_defaults())
    return res



def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def align_lists(list1, list2, aligned_place):
    aligned_elements = [list1[i] for i in aligned_place]
    list2_aligned = []
    from bisect import bisect_left

    left = 0
    for t in aligned_elements:
        idx = bisect_left(list2[left:], t)
        list2_aligned += list2[left:left+idx] + [t]
        left += idx + 1
    list2_aligned += list2[left:]
    return list2_aligned



def create_space_timesteps(steps=1000, forward_step=50, backward_step=50, align_place=[]):
    # assert forward_step % backward_step == 0 or backward_step % forward_step == 0

    frac_stride = (steps - 1) / (forward_step - 1)
    forward_step_list = []
    cur_idx = 0.0
    for _ in range(forward_step):
        forward_step_list.append(round(cur_idx))
        cur_idx += frac_stride

    frac_stride = (steps - 1) / (backward_step - 1)
    backward_step_list = []
    cur_idx = 0.0
    for _ in range(backward_step):
        backward_step_list.append(round(cur_idx))
        cur_idx += frac_stride

    backward_step_list = align_lists(forward_step_list, backward_step_list, align_place)

    return set(forward_step_list), set(backward_step_list)


def create_forward_backward_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    forward_timestep=None,
    backward_timestep=None,
    insert_step=None,
    align_place=[],
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if forward_timestep is None:
        forward_timestep = steps
    if backward_timestep is None:
        backward_timestep = steps
    if insert_step is None:
        insert_step = steps

    forward_step_list, backward_step_list = create_space_timesteps(insert_step, forward_timestep, backward_timestep, align_place)

    forward_diffusion = SpacedDiffusion(
        use_timesteps=forward_step_list,
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

    backward_diffusion = SpacedDiffusion(
        use_timesteps=backward_step_list,
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

    return forward_diffusion, backward_diffusion