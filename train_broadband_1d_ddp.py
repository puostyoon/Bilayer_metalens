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
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# DDP imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import lpips
import matplotlib.pyplot as plt

from lut import KeyedLUTSampler3D_HardGumbel_Complex
from utils.utils import *  # (Propagator, Aperture, fft_convolve2d, pado 등 프로젝트 유틸)
from model.DA_loss_functions import DA_loss
from pytorch_msssim import ms_ssim


# ============================================================
# DDP Setup Utilities
# ============================================================
def setup_ddp():
    """Initialize distributed training from environment variables set by torchrun."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process():
    return get_rank() == 0


def broadcast_tensor(tensor, src=0):
    """Broadcast a tensor from src rank to all other ranks."""
    if dist.is_initialized():
        dist.broadcast(tensor, src=src)
    return tensor


# ============================================================
# OpticsParams Module for DDP
# ============================================================
class OpticsParams(nn.Module):
    """
    Wraps the learnable optical parameters as nn.Parameters so they can be
    managed by DistributedDataParallel.
    """
    def __init__(self, Nr: int, K_lib: int, device):
        super().__init__()
        # shared across classes: (1,1,1,Nr)
        self.layer1_profile_raw = nn.Parameter(torch.randn((1, 1, 1, Nr), device=device))
        self.layer2_profile_raw = nn.Parameter(torch.randn((1, 1, 1, Nr), device=device))
        # orderless class logits along radius: (1,K,1,Nr)
        self.class_logits_profile_raw = nn.Parameter(torch.zeros((1, K_lib, 1, Nr), device=device))

    def forward(self):
        # Just return the parameters for use in the training step
        return self.layer1_profile_raw, self.layer2_profile_raw, self.class_logits_profile_raw


# ============================================================
# Rotational symmetry helpers
# ============================================================
def make_radial_sampling_grid(R: int, C: int, device, dtype=torch.float32, align_corners: bool = True):
    """
    Build a fixed grid for sampling a 1D radial profile (shape: (1,K,1,Nr))
    into a 2D rotationally symmetric map (shape: (1,K,R,C)) via grid_sample.

    We treat the 1D profile as an "image" of size (H=1, W=Nr).
    grid_sample expects grid coords in [-1,1] for (x,y).

    - x coord encodes radius r_norm in [0,1] mapped to [-1,1]
    - y coord is always 0 (since H=1)

    r_norm is computed on the R×C grid as distance-to-center normalized by
    inscribed circle radius (min((R-1)/2, (C-1)/2)). Outside is clamped.
    """
    yy = torch.arange(R, device=device, dtype=dtype).view(R, 1).expand(R, C)
    xx = torch.arange(C, device=device, dtype=dtype).view(1, C).expand(R, C)

    cy = (R - 1) / 2.0
    cx = (C - 1) / 2.0
    dy = yy - cy
    dx = xx - cx
    r = torch.sqrt(dx * dx + dy * dy)

    r_max = min(cy, cx)
    if r_max <= 0:
        r_max = 1.0

    r_norm = (r / r_max).clamp(0.0, 1.0)  # [0,1]

    # map [0,1] -> [-1,1] for x coordinate
    grid_x = (2.0 * r_norm - 1.0)

    # y=0 (center row) because profile H=1
    grid_y = torch.zeros_like(grid_x)

    grid = torch.stack([grid_x, grid_y], dim=-1)  # (R,C,2)
    grid = grid.unsqueeze(0)  # (1,R,C,2)
    return grid


def radial_profile_to_2d_map(profile_1d, radial_grid_2d, K_out: int, align_corners: bool = True):
    """
    profile_1d:
      - (1,1,1,Nr)  -> shared across classes (we expand to K_out)
      - OR (1,K,1,Nr) -> per-class profile (K must match K_out)

    radial_grid_2d: (1,R,C,2) from make_radial_sampling_grid

    returns: (1,K_out,R,C)
    """
    assert profile_1d.ndim == 4, f"profile_1d must be (1,*,1,Nr), got {profile_1d.shape}"
    Bp, Kp, Hp, Nr = profile_1d.shape
    assert Bp == 1 and Hp == 1, "profile_1d must be (1,K,1,Nr) or (1,1,1,Nr)"

    if Kp == 1 and K_out > 1:
        prof = profile_1d.expand(1, K_out, 1, Nr)   # share across classes
    else:
        assert Kp == K_out, f"profile K={Kp} must match K_out={K_out} (or be 1 to share)."
        prof = profile_1d

    # grid_sample: input (N,C,H,W), grid (N,H_out,W_out,2) -> (N,C,H_out,W_out)
    out = F.grid_sample(
        prof, radial_grid_2d,
        mode="nearest",
        align_corners=align_corners
    )
    return out  # (1,K_out,R,C)


# ============================================================
# Original training utilities (kept)
# ============================================================
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


def image_loss_increase_contrast(batch_data, complex_field, wvls, args, step, eval=False, verbose=True):
    param = args.param
    light = pado.Light(complex_field.shape, pitch=param.meta_pitch, wvl=wvls, field=complex_field, device=args.device)
    psf = compute_psf(wvls, light, args, 'SBL_ASM', offset=(0, 0), normalize=False)

    incident_light_intensity_sum = light.get_amplitude().sum(dim=(-2,-1), keepdim=True) * ((param.meta_pitch/param.camera_pitch)**2)

    image, _ = batch_data
    image = image.to(args.device)
    assert image.shape[0] == 1, f'Batch size should be 1. image shape {image.shape}'

    # RGB -> Grayscale using ALL channels
    if image.shape[1] == 3:
        image = (
            0.2989 * image[:, 0:1, ...] +
            0.5870 * image[:, 1:2, ...] +
            0.1140 * image[:, 2:3, ...]
        )
    elif image.shape[1] == 1:
        pass
    else:
        image = image.mean(dim=1, keepdim=True)

    if complex_field.shape[1] != image.shape[1]:
        psf = psf.permute(1, 0, 2, 3)
        incident_light_intensity_sum = incident_light_intensity_sum.permute(1, 0, 2, 3)
        image = image.repeat(psf.shape[0], 1, 1, 1)

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

    if verbose and is_main_process():
        print(
            f'step: {step}, wvls(nm): {[int(w*1e9) for w in wvls]} '
            f'loss_total: {loss.item():.6f} (orig {loss_original.item():.6f}, bright {loss_brightness_regularizer.item():.6f})'
        )

    if eval:
        os.makedirs(os.path.join(args.result_path, 'logged_far_image'), exist_ok=True)
        for ch in range(convolved_image.shape[0]):
            plt.figure(); plt.imshow(np.clip((convolved_image).detach().cpu().numpy()[ch, 0], 0, 1)); plt.colorbar()
            plt.savefig(os.path.join(args.result_path, 'logged_far_image', f'step_{step}_ch{ch}.png'), bbox_inches='tight')
            plt.clf(); plt.close()

    return loss


def transmittance_penalty_all_wvls(
    complex_field_dict,
    wvls,
    trans_min: float,
    metric: str = "intensity",
    penalty_type: str = "hinge_l2"
):
    penalties = []
    for w in wvls:
        key = int(w * 1e9)
        if key not in complex_field_dict:
            continue
        z = complex_field_dict[key]

        if metric == "amplitude":
            t_mean = z.abs().mean()
        else:
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


def train_step_compute_loss(
    args, step, batch_data, lut_sampler,
    layer1_profile_raw, layer2_profile_raw, class_logits_profile_raw,
    radial_grid_2d, K_lib, eval, wvls, verbose=True
):
    """
    Rotational symmetry version. Returns loss tensor (for backward) and logits_2d.
    Does NOT call backward() - that is handled by the caller for DDP no_sync control.

    layer1_profile_raw: (1,1,1,Nr)  (shared across classes) OR (1,K,1,Nr) (per-class)
    layer2_profile_raw: (1,1,1,Nr)  (shared across classes) OR (1,K,1,Nr)
    class_logits_profile_raw: (1,K,1,Nr)  (orderless class logits along radius)

    We lift them to:
      layer1_raw_2d: (1,K,R,C)
      layer2_raw_2d: (1,K,R,C)
      class_logits_2d: (1,K,R,C)
    """

    # ----- lift 1D -> 2D rotationally symmetric maps -----
    layer1_raw_2d = radial_profile_to_2d_map(layer1_profile_raw, radial_grid_2d, K_out=K_lib, align_corners=True)
    layer2_raw_2d = radial_profile_to_2d_map(layer2_profile_raw, radial_grid_2d, K_out=K_lib, align_corners=True)
    class_logits_2d = radial_profile_to_2d_map(class_logits_profile_raw, radial_grid_2d, K_out=K_lib, align_corners=True)

    # ----- (1) per-class xy in [-1,1] -----
    x_all = 2.0 * torch.sigmoid(layer2_raw_2d) - 1.0  # (1,K,R,C)
    y_all = 2.0 * torch.sigmoid(layer1_raw_2d) - 1.0  # (1,K,R,C)

    xy = torch.stack([x_all, y_all], dim=2)           # (1,K,2,R,C)
    B, K, _, H, W = xy.shape
    grid = xy.reshape(B, 2*K, H, W)                   # (1,2K,R,C)

    # ----- (2) logits for orderless class selection -----
    logits = class_logits_2d                           # (1,K,R,C)

    # ----- (3) LUT sampling (true complex) -----
    keys_nm = [int(w * 1e9) for w in wvls]
    complex_field_dict = lut_sampler(keys=keys_nm, grid=grid, class_logits=logits)

    complex_field = torch.cat([complex_field_dict[k] for k in keys_nm], dim=1)  # (1,len(wvls),R,C)

    # ----- (4) image loss -----
    loss_img = image_loss_increase_contrast(batch_data, complex_field, wvls, args, step, eval=eval, verbose=verbose)

    # ----- (5) optional transmittance penalty -----
    if args.use_transmittance_penalty:
        loss_trans = transmittance_penalty_all_wvls(
            complex_field_dict, wvls,
            trans_min=args.transmittance_min,
            metric=args.transmittance_metric,
            penalty_type=args.transmittance_penalty_type,
        )
        loss_total = loss_img + args.transmittance_penalty_weight * loss_trans
        if verbose and is_main_process():
            print(f"  trans_penalty: {loss_trans.item():.6f} (w={args.transmittance_penalty_weight})")
    else:
        loss_total = loss_img

    # Return loss tensor (not detached) for backward, and logits for logging
    return loss_total, class_logits_2d


def batch_iterator(arr, batch_size):
    n = len(arr)
    for i in range(0, n, batch_size):
        yield arr[i:i+batch_size]


def shard_wavelengths(wvls, rank, world_size):
    """
    Shard wavelengths across ranks using strided partitioning.
    Each rank gets wvls[rank::world_size].
    Returns the local subset and total global count.
    """
    wvls_local = wvls[rank::world_size]
    return wvls_local, len(wvls)


def train(args, local_rank):
    """
    DDP-enabled training function.
    - Wavelengths are sharded across GPUs
    - Same image batch is used by all ranks (broadcast from rank 0)
    - Gradient accumulation with no_sync() for efficiency
    """
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    args.device = device

    # Only main process writes logs
    writer = None
    if is_main_process():
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

    # Use same shuffle order across all ranks by setting same seed for dataloader
    # This ensures all ranks process the same images when we broadcast
    g = torch.Generator()
    g.manual_seed(args.seed)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
        drop_last=True, generator=g
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )

    # losses
    args.MSE_criterion = nn.MSELoss().to(device)
    args.l1_criterion = nn.MSELoss().to(device)
    args.perc_criterion = None
    if args.use_perc_loss:
        args.perc_criterion = lpips.LPIPS(net='alex').to(device)

    # ----- Load LUTs -----
    cycy = torch.tensor(np.load(args.cycy), device=device).to(torch.complex64)
    cysq = torch.tensor(np.load(args.cysq), device=device).to(torch.complex64)
    sqcy = torch.tensor(np.load(args.sqcy), device=device).to(torch.complex64)
    sqsq = torch.tensor(np.load(args.sqsq), device=device).to(torch.complex64)

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
        use_hard=False,
        hard_eval=True,
    ).to(device)

    # ----- rotational symmetry: precompute radial grid once -----
    radial_grid_2d = make_radial_sampling_grid(
        R=param.R, C=param.C,
        device=device,
        dtype=torch.float32,
        align_corners=True
    )  # (1,R,C,2)

    # ----- learnable parameters wrapped in nn.Module for DDP -----
    Nr = args.radial_bins if args.radial_bins is not None else param.C
    optics_params = OpticsParams(Nr=Nr, K_lib=K_lib, device=device)

    # Wrap in DDP
    optics_params_ddp = DDP(optics_params, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.AdamW(
        [
            {"params": [optics_params.layer1_profile_raw, optics_params.layer2_profile_raw], "lr": args.optics_layer_lr},
            {"params": [optics_params.class_logits_profile_raw], "lr": args.optics_class_lr},
        ],
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.T_max, eta_min=min(args.optics_layer_lr, args.optics_class_lr) * 0.01
    )

    # ----- Wavelength sharding setup -----
    # Each rank processes a disjoint subset of wavelengths
    all_training_wvls = list(param.training_wvls)
    local_training_wvls, total_wvls = shard_wavelengths(all_training_wvls, rank, world_size)

    if is_main_process():
        print(f"[DDP] World size: {world_size}")
        print(f"[DDP] Total wavelengths: {total_wvls}, per-rank: ~{len(local_training_wvls)}")

    total_step = 0
    eval_minimum_loss = float("inf")

    epoch_iter = tqdm(range(args.n_epochs), desc="epoch") if is_main_process() else range(args.n_epochs)

    for epoch in epoch_iter:
        # Ensure all ranks have same dataloader order
        trainloader.dataset  # trigger any lazy init

        train_iter = tqdm(trainloader, desc="iter", leave=False) if is_main_process() else trainloader

        for it, batch_data in enumerate(train_iter):
            # ----- Broadcast image batch from rank 0 to all ranks -----
            # This ensures all ranks use the exact same image
            images, labels = batch_data
            images = images.to(device)
            labels = labels.to(device)
            broadcast_tensor(images, src=0)
            broadcast_tensor(labels, src=0)
            batch_data = (images, labels)

            progress = min(total_step / max(args.T_max, 1.0), 1.0)

            hard_start = getattr(args, "hard_start", 0.70)
            lut_sampler.use_hard = (progress >= hard_start)

            warm_frac = getattr(args, "warm_frac", 0.60)
            if progress <= warm_frac:
                tau = args.tau_start
            else:
                t01 = (progress - warm_frac) / max(1.0 - warm_frac, 1e-12)
                tau = args.tau_end + 0.5 * (args.tau_start - args.tau_end) * (1.0 + math.cos(math.pi * t01))
            tau = max(float(tau), 1e-4)
            lut_sampler.tau = tau

            optimizer.zero_grad(set_to_none=True)

            # ----- Wavelength microbatching with DDP no_sync -----
            # Each rank processes its local wavelength subset
            # Use no_sync() for all but the last microbatch to avoid expensive allreduce per microbatch
            local_wvl_batches = list(batch_iterator(local_training_wvls, param.wvl_batch_size))
            num_local_batches = len(local_wvl_batches)

            step_loss_accum = 0.0
            last_logits_2d = None

            # Get references to the underlying parameters
            layer1_profile_raw = optics_params.layer1_profile_raw
            layer2_profile_raw = optics_params.layer2_profile_raw
            class_logits_profile_raw = optics_params.class_logits_profile_raw

            for batch_idx, wvls in enumerate(local_wvl_batches):
                is_last_batch = (batch_idx == num_local_batches - 1)

                # Use no_sync for all batches except the last one
                # On the last batch, allow DDP to sync gradients across ranks
                sync_context = nullcontext() if is_last_batch else optics_params_ddp.no_sync()

                with sync_context:
                    loss, logits_2d = train_step_compute_loss(
                        args, total_step, batch_data, lut_sampler,
                        layer1_profile_raw, layer2_profile_raw, class_logits_profile_raw,
                        radial_grid_2d, K_lib,
                        eval=False, wvls=wvls,
                        verbose=(is_main_process())  # only print first batch on rank 0
                    )
                    last_logits_2d = logits_2d

                    # Scale loss for proper gradient averaging across all wavelengths globally
                    # Each microbatch contributes proportionally to the total
                    num_global_microbatches = num_local_batches * world_size
                    scaled_loss = loss / num_global_microbatches

                    scaled_loss.backward()
                    step_loss_accum += loss.detach().item()

            # After backward on last batch, DDP has synchronized gradients
            optimizer.step()
            scheduler.step()

            # ----- Aggregate loss across ranks for logging -----
            loss_tensor = torch.tensor([step_loss_accum], device=device)
            if dist.is_initialized():
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            global_step_loss = loss_tensor.item() / world_size

            # ----- logging (only main process) -----
            if is_main_process() and total_step % args.log_freq == 0 and total_step > 0:
                with torch.no_grad():
                    if last_logits_2d is None:
                        last_logits_2d = radial_profile_to_2d_map(class_logits_profile_raw, radial_grid_2d, K_out=K_lib, align_corners=True)

                    probs = F.softmax(last_logits_2d / max(lut_sampler.tau, 1e-8), dim=1)
                    probs_mean = probs.mean(dim=(-2, -1)).squeeze(0)
                    for k in range(K_lib):
                        writer.add_scalar(f'class/prob_mean_{k}', probs_mean[k].item(), total_step)

                    writer.add_scalar("train/loss_step_accum", global_step_loss, total_step)
                    writer.add_scalar("train/tau", lut_sampler.tau, total_step)
                    writer.add_scalar("train/use_hard", float(lut_sampler.use_hard), total_step)

                    # eval (only on main process)
                    eval_loss = 0.0
                    lut_sampler.eval()
                    for vb, vdata in enumerate(testloader):
                        vimages, vlabels = vdata
                        vimages = vimages.to(device)
                        vdata = (vimages, vlabels)
                        # Use all wavelengths for eval on main process
                        for wvls in batch_iterator(all_training_wvls, param.wvl_batch_size):
                            eloss, _ = train_step_compute_loss(
                                args, total_step, vdata, lut_sampler,
                                layer1_profile_raw, layer2_profile_raw, class_logits_profile_raw,
                                radial_grid_2d, K_lib,
                                eval=True, wvls=wvls, verbose=False
                            )
                            eval_loss += eloss.detach().item()
                    lut_sampler.train()

                    writer.add_scalar("eval/loss", eval_loss, total_step)

                    if eval_loss < eval_minimum_loss:
                        eval_minimum_loss = eval_loss
                        torch.save(layer1_profile_raw, os.path.join(args.result_path, 'layer1_profile_min_eval.pt'))
                        torch.save(layer2_profile_raw, os.path.join(args.result_path, 'layer2_profile_min_eval.pt'))
                        torch.save(class_logits_profile_raw, os.path.join(args.result_path, 'class_logits_profile_min_eval.pt'))

            # save checkpoint (only main process)
            if is_main_process() and total_step % args.save_freq == 0 and total_step > 0:
                torch.save(layer1_profile_raw, os.path.join(args.result_path, f'layer1_profile_{total_step:06d}.pt'))
                torch.save(layer2_profile_raw, os.path.join(args.result_path, f'layer2_profile_{total_step:06d}.pt'))
                torch.save(class_logits_profile_raw, os.path.join(args.result_path, f'class_logits_profile_{total_step:06d}.pt'))

            total_step += 1

            # Barrier to keep all ranks in sync
            if dist.is_initialized():
                dist.barrier()

    if writer is not None:
        writer.close()


def main():
    parser = argparse.ArgumentParser(
        description='PSF based Obstruction-free Metasurface training with DDP (rotational symmetry param)',
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
    parser.add_argument('--optics_layer_lr', default=0.1, type=float)
    parser.add_argument('--optics_class_lr', default=0.05, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)

    parser.add_argument('--T_max', default=5000, type=float)
    parser.add_argument('--tau_start', default=0.5, type=float)
    parser.add_argument('--tau_end', default=0.5, type=float)
    parser.add_argument('--hard_start', default=0.0, type=float)
    parser.add_argument('--warm_frac', default=0.0, type=float)

    parser.add_argument('--log_freq', default=30, type=int)
    parser.add_argument('--save_freq', default=400, type=int)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--phase_init', default='zero', type=str)
    parser.add_argument('--batch_size', default=1, type=int)

    # rotational symmetry
    parser.add_argument('--radial_bins', default=None, type=int,
                        help="Nr for radial profile. If None, uses param.C.")

    # Loss flags
    parser.add_argument('--use_perc_loss', action="store_true")
    parser.add_argument('--use_ssim_loss', action="store_true")
    parser.add_argument('--use_da_loss', action="store_true")
    parser.add_argument('--l1_loss_weight', default=1.0, type=float)
    parser.add_argument('--perc_loss_weight', default=1.0, type=float)
    parser.add_argument('--ssim_loss_weight', default=1.0, type=float)
    parser.add_argument('--da_loss_weight', default=1.0, type=float)
    parser.add_argument('--brightness_regularizer_weight', default=1.0, type=float)

    # Transmittance penalty (ALL wavelengths)
    parser.add_argument('--use_transmittance_penalty', action="store_true")
    parser.add_argument('--transmittance_penalty_weight', default=1.0, type=float)
    parser.add_argument('--transmittance_min', default=0.25, type=float)
    parser.add_argument('--transmittance_metric', default="intensity", type=str, choices=["amplitude", "intensity"])
    parser.add_argument('--transmittance_penalty_type', default="hinge_l2", type=str, choices=["hinge_l1", "hinge_l2"])

    args = parser.parse_args()

    # ----- Initialize DDP -----
    local_rank = setup_ddp()

    # Set seeds for reproducibility (same across all ranks)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    param = SourceFileLoader("param", args.param_file).load_module()

    # phase init (kept as-is) - use CPU first, will move to device in train()
    if args.phase_init == 'random':
        param.phase_init = torch.rand(param.phase_init.shape) * 10
    elif args.phase_init == 'fresnel':
        param.phase_init = RefractiveLens(param.phase_init.shape, param.meta_pitch, param.focal_length, param.meta_wvl, 'cpu').get_phase_change()
    else:
        param.phase_init = torch.zeros(param.phase_init.shape)

    # Only main process creates directories and saves config
    if is_main_process():
        if not os.path.exists(args.result_path):
            os.makedirs(args.result_path)
        else:
            raise Exception("The directory already exists!")

        with open(os.path.join(args.result_path, 'args.json'), "w") as f:
            json.dump(vars(args), f, indent=4, sort_keys=False)
        shutil.copy(args.param_file, args.result_path)

    # Wait for main process to create directories
    if dist.is_initialized():
        dist.barrier()

    args.param = param

    try:
        train(args, local_rank)
    finally:
        cleanup_ddp()


if __name__ == '__main__':
    main()
