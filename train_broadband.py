import os
import sys
import time
import math
import json
import shutil
import argparse
import numpy as np
from tqdm.auto import tqdm
from importlib.machinery import SourceFileLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import lpips
import matplotlib.pyplot as plt

from lut import KeyedLUTSampler3D_HardGumbel_Complex
from utils.utils import *  # (Propagator, Aperture, fft_convolve2d, pado 등 프로젝트 유틸)
from model.DA_loss_functions import DA_loss
from pytorch_msssim import ms_ssim


def compute_psf(wvls, light, args, propagator, offset=(0, 0), normalize=True):
    param = args.param
    prop = Propagator(propagator)
    dim = light.field.shape
    aperture = Aperture(dim, param.meta_pitch, param.aperture_diamter, param.aperture_shape, wvls, args.device)
    light = aperture.forward(light.clone())

    light_prop = prop.forward(light, param.sensor_dist, offset=offset, linear=args.propagator_linear_convolution)
    psf = light_prop.get_intensity()

    psf = F.interpolate(psf, scale_factor=light_prop.pitch/(param.camera_pitch), mode=args.resizing_method)
    if normalize:
        psf = psf / (torch.sum(psf, dim=(-2,-1), keepdim=True) + 1e-8)
    return psf


def image_loss_increase_contrast(batch_data, complex_field, wvls, args, step, eval=False):
    """
    기존 함수는 내부에서 loss.backward()를 해버려서 trans penalty 같은 추가 loss를 더하기가 어려움.
    -> 여기서는 loss tensor를 반환만 하고 backward는 train_step에서 1번만 수행.
    """
    param = args.param
    light = pado.Light(complex_field.shape, pitch=param.meta_pitch, wvl=wvls, field=complex_field, device=args.device)
    psf = compute_psf(wvls, light, args, 'SBL_ASM', offset=(0, 0), normalize=False)

    # brightness regularize
    incident_light_intensity_sum = light.get_amplitude().sum(dim=(-2,-1), keepdim=True) * ((param.meta_pitch/param.camera_pitch)**2)

    image, _ = batch_data
    image = image.to(args.device)
    assert image.shape[0] == 1, f'Batch size should be 1. image shape {image.shape}'

    # # RGB -> Grayscale using ALL channels
    if image.shape[1] == 3:
        # ITU-R BT.601 luma (standard luminance)
        image = (
            0.2989 * image[:, 0:1, ...] +
            0.5870 * image[:, 1:2, ...] +
            0.1140 * image[:, 2:3, ...]
        )
    elif image.shape[1] == 1:
        # already grayscale
        pass
    else:
        # fallback: average over channels if unexpected channel count
        image = image.mean(dim=1, keepdim=True)

    # match wavelength-batch dimension
    if complex_field.shape[1] != image.shape[1]:
        psf = psf.permute(1, 0, 2, 3)  # (B,1,H,W)  (B == #wvls in this batch)
        incident_light_intensity_sum = incident_light_intensity_sum.permute(1, 0, 2, 3)
        # repeat grayscale image for each wavelength "batch"
        image = image.repeat(psf.shape[0], 1, 1, 1)  # (B,1,H,W)

    convolved_image = fft_convolve2d(image, psf/psf.sum(dim=(-2,-1), keepdim=True))
    convolved_image_brightness_regularizer = fft_convolve2d(image, psf/incident_light_intensity_sum)

    loss_original = args.l1_loss_weight * args.MSE_criterion(convolved_image, image)
    loss_brightness_regularizer = args.l1_loss_weight * args.MSE_criterion(convolved_image_brightness_regularizer, image)

    if args.use_ssim_loss:
        loss_original = loss_original + args.ssim_loss_weight * (1 - ms_ssim(convolved_image, image))
        loss_brightness_regularizer = loss_brightness_regularizer + args.ssim_loss_weight * (1 - ms_ssim(convolved_image_brightness_regularizer, image))

    if args.use_da_loss:
        loss_original = loss_original + args.da_loss_weight * DA_loss(convolved_image, image, args, 'segmentation')
        loss_brightness_regularizer = loss_brightness_regularizer + args.da_loss_weight * DA_loss(convolved_image_brightness_regularizer, image, args, 'segmentation')

    if args.use_perc_loss:
        loss_original = loss_original + args.perc_loss_weight * args.perc_criterion(convolved_image, image).mean()
        loss_brightness_regularizer = loss_brightness_regularizer + args.perc_loss_weight * args.perc_criterion(convolved_image_brightness_regularizer, image).mean()

    loss = loss_original + args.brightness_regularizer_weight * loss_brightness_regularizer

    print(
        f'step: {step}, wvls(nm): {[int(w*1e9) for w in wvls]} '
        f'loss_total: {loss.item():.6f} (orig {loss_original.item():.6f}, bright {loss_brightness_regularizer.item():.6f})'
    )

    if eval:
        # 간단 로깅 (필요하면 유지/확장)
        os.makedirs(os.path.join(args.result_path, 'logged_psf'), exist_ok=True)
        os.makedirs(os.path.join(args.result_path, 'logged_far_image'), exist_ok=True)
        for ch in range(convolved_image.shape[0]):
            plt.figure(); plt.imshow(np.clip((convolved_image).detach().cpu().numpy()[ch, 0], 0, 1)); plt.colorbar()
            plt.savefig(os.path.join(args.result_path, 'logged_far_image', f'step_{step}_ch{ch}.png'), bbox_inches='tight')
            plt.clf(); plt.close()

    return loss


def transmittance_penalty_all_wvls(
    complex_field_dict,  # {wvl_nm_int: (B,1,H,W) complex}
    wvls,                # list of wvl in meters
    trans_min: float,
    metric: str = "intensity",     # "amplitude" or "intensity"
    penalty_type: str = "hinge_l2" # "hinge_l1" or "hinge_l2"
):
    """
    모든 wvl에 대해 평균 transmittance가 trans_min 아래면 penalty.
    metric:
      - amplitude: mean(|E|)
      - intensity: mean(|E|^2)
    """
    penalties = []
    for w in wvls:
        key = int(w * 1e9)
        if key not in complex_field_dict:
            continue
        z = complex_field_dict[key]  # (B,1,H,W) complex

        if metric == "amplitude":
            t_mean = z.abs().mean()
        else:  # intensity
            t_mean = (z.abs().pow(2)).mean()

        if penalty_type == "hinge_l1":
            p = F.relu(trans_min - t_mean)
        else:
            p = F.relu(trans_min - t_mean).pow(2)

        penalties.append(p)

    if len(penalties) == 0:
        device = next(iter(complex_field_dict.values())).device
        return torch.tensor(0.0, device=device)

    return torch.stack(penalties).mean()


def train_step_backward(
    args, step, batch_data, lut_sampler,
    layer1_indices, layer2_indices, class_logits,
    eval, wvls
):
    """
    layer1_indices: (1,K,R,C)
    layer2_indices: (1,K,R,C)
    class_logits  : (1,K,R,C)
    """

    # ----- (1) per-class xy in [-1,1] -----
    # x = layer2, y = layer1 (너 코드 컨벤션 유지)
    x_all = 2.0 * torch.sigmoid(layer2_indices) - 1.0  # (1,K,H,W)
    y_all = 2.0 * torch.sigmoid(layer1_indices) - 1.0  # (1,K,H,W)

    # interleave -> (B,2K,H,W): [x0,y0,x1,y1,...]
    xy = torch.stack([x_all, y_all], dim=2)            # (B,K,2,H,W)
    B, K, _, H, W = xy.shape
    grid = xy.reshape(B, 2*K, H, W)

    # ----- (2) logits shape 맞추기 -----
    logits = class_logits
    if logits.shape[0] != B:
        logits = logits.expand(B, -1, -1, -1)

    # ----- (3) LUT sampling (true complex) -----
    keys_nm = [int(w * 1e9) for w in wvls]
    complex_field_dict = lut_sampler(keys=keys_nm, grid=grid, class_logits=logits)

    # concat wavelengths into channel dim (B, len(wvls), H, W)
    complex_field = torch.cat([complex_field_dict[k] for k in keys_nm], dim=1)

    # ----- (4) image loss -----
    loss_img = image_loss_increase_contrast(batch_data, complex_field, wvls, args, step, eval=eval)

    # ----- (5) optional transmittance penalty (ALL wvls) -----
    if args.use_transmittance_penalty:
        loss_trans = transmittance_penalty_all_wvls(
            complex_field_dict, wvls,
            trans_min=args.transmittance_min,
            metric=args.transmittance_metric,
            penalty_type=args.transmittance_penalty_type,
        )
        loss_total = loss_img + args.transmittance_penalty_weight * loss_trans
        print(f"  trans_penalty: {loss_trans.item():.6f} (w={args.transmittance_penalty_weight})")
    else:
        loss_total = loss_img

    # Entropy bonus (to encourage wider class usage)
    # probs
    p = F.softmax(class_logits / max(lut_sampler.tau, 1e-8), dim=1)  # (1,K,H,W)

    # per-pixel entropy (0 ~ logK)
    ent = -(p * (p + 1e-12).log()).sum(dim=1).mean()

    # entropy를 키우고 싶으면 loss에서 "빼야" 함
    loss_total = loss_img - args.entropy_weight * ent
    print(f"entropy bonus: {ent.item()}")

    if not eval:
        loss_total.backward()

    return float(loss_total.detach().item())


def batch_iterator(arr, batch_size):
    n = len(arr)
    for i in range(0, n, batch_size):
        yield arr[i:i+batch_size]


def train(args):
    writer = SummaryWriter(log_dir=os.path.join(args.result_path, 'runs'))
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    param = args.param

    transform_train = transforms.Compose([
        transforms.Resize([param.img_res, param.img_res]),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize([param.img_res, param.img_res]),
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.ImageFolder(args.training_dir, transform=transform_train)
    testset = torchvision.datasets.ImageFolder(args.validation_dir, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=1, pin_memory=True,
        persistent_workers=True, prefetch_factor=1,
        drop_last=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=1, pin_memory=True,
        persistent_workers=True, prefetch_factor=1
    )

    # losses
    args.MSE_criterion = nn.MSELoss().to(args.device)
    args.l1_criterion = nn.MSELoss().to(args.device)
    args.perc_criterion = None
    if args.use_perc_loss:
        args.perc_criterion = lpips.LPIPS(net='alex').to(args.device)

    # ----- Load LUTs -----
    cycy = torch.tensor(np.load(args.cycy), device=args.device).to(torch.complex64)
    cysq = torch.tensor(np.load(args.cysq), device=args.device).to(torch.complex64)
    sqcy = torch.tensor(np.load(args.sqcy), device=args.device).to(torch.complex64)
    sqsq = torch.tensor(np.load(args.sqsq), device=args.device).to(torch.complex64)

    lut_dict = {
        int(wvl * 1e9): torch.stack(
            [cycy[..., wvl_idx], cysq[..., wvl_idx], sqcy[..., wvl_idx], sqsq[..., wvl_idx]],
            dim=0
        )
        for (wvl_idx, wvl) in enumerate(param.full_broadband_wvls)
    }
    K_lib = next(iter(lut_dict.values())).shape[0]

    lut_sampler = KeyedLUTSampler3D_HardGumbel_Complex(
        lut_dict=lut_dict,
        align_corners=True,
        tau=args.tau_start,
        use_hard=False,              # 스케줄로 바꿈
        hard_eval=True,
    ).to(args.device)

    # ----- learnable parameters -----
    # per-class xy indices (intialize two layers identically)
    layer1_indices = torch.randn((1, K_lib, param.R, param.C), device=args.device, requires_grad=True)
    layer2_indices = layer1_indices.detach().clone().requires_grad_(True)

    # orderless class selection logits
    class_logits = torch.zeros((1, K_lib, param.R, param.C), device=args.device, requires_grad=True)

    # optimizer (param group 분리)
    optimizer = optim.AdamW(
        [
            {"params": [layer1_indices, layer2_indices], "lr": args.optics_layer_lr},
            {"params": [class_logits], "lr": args.optics_class_lr},
        ],
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.T_max, eta_min=min(args.optics_layer_lr, args.optics_class_lr) * 0.01
    )

    total_step = 0
    eval_minimum_loss = float("inf")

    for epoch in tqdm(range(args.n_epochs), desc="epoch"):
        for it, batch_data in enumerate(tqdm(trainloader, desc="iter", leave=False)):
            # 0..1 progress
            progress = min(total_step / max(args.T_max, 1.0), 1.0)

            # ----- safe soft->hard switch -----
            hard_start = getattr(args, "hard_start", 0.70)
            lut_sampler.use_hard = (progress >= hard_start)

            # ----- tau schedule (warm-hold then cosine) -----
            warm_frac = getattr(args, "warm_frac", 0.60)
            if progress <= warm_frac:
                tau = args.tau_start
            else:
                t01 = (progress - warm_frac) / max(1.0 - warm_frac, 1e-12)
                tau = args.tau_end + 0.5 * (args.tau_start - args.tau_end) * (1.0 + math.cos(math.pi * t01))
            tau = max(float(tau), 1e-4)
            lut_sampler.tau = tau

            # ----- train over wavelength mini-batches -----
            optimizer.zero_grad(set_to_none=True)

            step_loss_accum = 0.0
            for wvls in batch_iterator(param.training_wvls, args.wvl_batch_size):
                step_loss = train_step_backward(
                    args, total_step, batch_data, lut_sampler,
                    layer1_indices, layer2_indices, class_logits,
                    eval=False, wvls=wvls
                )
                step_loss_accum += step_loss

            optimizer.step()
            scheduler.step()

            # ----- logging -----
            if total_step % args.log_freq == 0 and total_step > 0:
                with torch.no_grad():
                    # class prob stats
                    probs = F.softmax(class_logits / max(lut_sampler.tau, 1e-8), dim=1)  # (1,K,H,W)
                    probs_mean = probs.mean(dim=(-2, -1)).squeeze(0)                     # (K,)
                    for k in range(K_lib):
                        writer.add_scalar(f'class/prob_mean_{k}', probs_mean[k].item(), total_step)

                    writer.add_scalar("train/loss_step_accum", step_loss_accum, total_step)
                    writer.add_scalar("train/tau", lut_sampler.tau, total_step)
                    writer.add_scalar("train/use_hard", float(lut_sampler.use_hard), total_step)

                    # eval
                    eval_loss = 0.0
                    lut_sampler.eval()
                    for vb, vdata in enumerate(testloader):
                        for wvls in batch_iterator(param.training_wvls, args.wvl_batch_size):
                            eval_loss += train_step_backward(
                                args, total_step, vdata, lut_sampler,
                                layer1_indices, layer2_indices, class_logits,
                                eval=True, wvls=wvls
                            )
                    lut_sampler.train()

                    writer.add_scalar("eval/loss", eval_loss, total_step)

                    if eval_loss < eval_minimum_loss:
                        eval_minimum_loss = eval_loss
                        torch.save(layer1_indices, os.path.join(args.result_path, 'layer1_indices_min_eval.pt'))
                        torch.save(layer2_indices, os.path.join(args.result_path, 'layer2_indices_min_eval.pt'))
                        torch.save(class_logits,  os.path.join(args.result_path, 'class_logits_min_eval.pt'))

            # save checkpoint
            if total_step % args.save_freq == 0 and total_step > 0:
                torch.save(layer1_indices, os.path.join(args.result_path, f'layer1_indices_{total_step:06d}.pt'))
                torch.save(layer2_indices, os.path.join(args.result_path, f'layer2_indices_{total_step:06d}.pt'))
                torch.save(class_logits,  os.path.join(args.result_path, f'class_logits_{total_step:06d}.pt'))

            total_step += 1

            # clear (optional)
            if os.name != "nt":
                os.system("clear")

    writer.close()


def main():
    parser = argparse.ArgumentParser(
        description='PSF based Obstruction-free Metasurface training',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Paths
    parser.add_argument('--result_path', default='./example/asset/ckpt/metasurface', type=str)
    parser.add_argument('--eval_path', default="dataset/eval", type=str)
    parser.add_argument('--training_dir', default="dataset/eval", type=str)
    parser.add_argument('--validation_dir', default="dataset/eval", type=str)
    parser.add_argument('--sqsq', default='./library/sqsq.npy', type=str)
    parser.add_argument('--sqcy', default='./library/sqcy.npy', type=str)
    parser.add_argument('--cycy', default='./library/cycy.npy', type=str)
    parser.add_argument('--cysq', default='./library/cysq.npy', type=str)
    parser.add_argument('--param_file', default='./example/asset/config/param_MV_1600_metasurface.py', type=str)

    # Simulation
    parser.add_argument('--propagator', default='SBL_ASM', type=str)
    parser.add_argument('--propagator_linear_convolution', action="store_true")
    parser.add_argument('--resizing_method', default='area', type=str)
    parser.add_argument('--normalizing_method', default='original', type=str)

    # Training config
    parser.add_argument('--n_epochs', default=1, type=int)
    parser.add_argument('--wvl_batch_size', default=14, type=int, help="Number of wavelengths per microbatch per rank.")
    parser.add_argument('--optics_layer_lr', default=0.1, type=float)
    parser.add_argument('--optics_class_lr', default=0.05, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)

    parser.add_argument('--T_max', default=5000, type=float)
    parser.add_argument('--tau_start', default=5.0, type=float)
    parser.add_argument('--tau_end', default=0.7, type=float)
    parser.add_argument('--hard_start', default=0.7, type=float, help="When to start hard one-hot mapping for forward")
    parser.add_argument('--warm_frac', default=0.6, type=float, help="When to start tau scheduling in softmax (see lut.py)")

    parser.add_argument('--log_freq', default=30, type=int)
    parser.add_argument('--save_freq', default=400, type=int)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--phase_init', default='zero', type=str)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--batch_size', default=1, type=int)

    # Loss flags
    parser.add_argument('--use_perc_loss', action="store_true")
    parser.add_argument('--use_ssim_loss', action="store_true")
    parser.add_argument('--use_da_loss', action="store_true")
    parser.add_argument('--l1_loss_weight', default=1.0, type=float)
    parser.add_argument('--perc_loss_weight', default=1.0, type=float)
    parser.add_argument('--ssim_loss_weight', default=1.0, type=float)
    parser.add_argument('--da_loss_weight', default=1.0, type=float)
    parser.add_argument('--brightness_regularizer_weight', default=1.0, type=float)
    parser.add_argument('--entropy_weight', default=1.0, type=float)

    # Transmittance penalty (ALL wavelengths)
    parser.add_argument('--use_transmittance_penalty', action="store_true")
    parser.add_argument('--transmittance_penalty_weight', default=1.0, type=float)
    parser.add_argument('--transmittance_min', default=0.25, type=float)
    parser.add_argument('--transmittance_metric', default="intensity", type=str, choices=["amplitude", "intensity"])
    parser.add_argument('--transmittance_penalty_type', default="hinge_l2", type=str, choices=["hinge_l1", "hinge_l2"])

    args = parser.parse_args()

    # seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load param
    param = SourceFileLoader("param", args.param_file).load_module()

    # phase init (kept as-is)
    if args.phase_init == 'random':
        param.phase_init = torch.rand(param.phase_init.shape, device=args.device) * 10
    elif args.phase_init == 'fresnel':
        param.phase_init = RefractiveLens(param.phase_init.shape, param.meta_pitch, param.focal_length, param.meta_wvl, args.device).get_phase_change()
    else:
        param.phase_init = torch.zeros(param.phase_init.shape, device=args.device)

    # save settings
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    else:
        raise Exception("The directory already exists!")

    with open(os.path.join(args.result_path, 'args.json'), "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=False)
    shutil.copy(args.param_file, args.result_path)

    args.param = param
    train(args)


if __name__ == '__main__':
    main()
