import pado
from pado.light import * 
from pado.optical_element import *
from pado.propagator import *
import torch
import torch.fft
import numpy as np

import os
import random
from matplotlib import pyplot   
import cv2
import shutil
import json
from glob import glob
from datetime import datetime
import skimage

from importlib.machinery import SourceFileLoader
from utils.Dirt import compute_dirt

# Image and PSF Manipulation

def sample_psf(psf, sample_ratio):
    if sample_ratio == 1:
        return psf
    else:
        return torch.nn.AvgPool2d(sample_ratio, stride=sample_ratio)(psf)
    

def compute_psf_arbitrary_prop(wvl, depth, doe, args, propagator='Fraunhofer', use_lens=True, offset=(0, 0), variable_offset_indices=None, theta=(0, 0), pad=True, full_psf=False, normalize=True):
    '''simulate depth based psf.
    Args:
        depth: propagation distance
        propagator: propagator defined in pado library. Fresenl, Fraunhofer, Rayleigh-Sommerfeld (RS), ASM, or BL_ASM
        use_lens: If True, DOE is placed in front of the lens.
        offset: (offset_y, offset_x) tuple. metric is meter. coordinate system follows matrix indexing ((0,0) is the top left corner.)
        theta: incident angle (theta_y, theta_x) of the point light source
        pad: If True, pad or unpad the psf size to be same as image size
        full_psf: If True, build full PSF which is impossible during training

    Note that Fresenl propagation with propagation distance focal length light after refractive lens is
    equal to the Fraunhofer propagation with propagation distance focal length.
    '''
    param = args.param
    prop = Propagator(propagator)
    dim = (1,1,param.R, param.C)

    # compute compute (dy, dx) for set_spherical_light function
    dy = np.tan(np.deg2rad(theta[0])) * (depth.numpy())
    dx = np.tan(np.deg2rad(theta[1])) * (depth.numpy())
    
    # set incident point light source
    light = Light(dim, param.DOE_pitch, wvl, device=args.device)
    light.set_spherical_light(depth.numpy(), dx=dx, dy=dy)

    if args.constant_wvl_phase:
        doe.wvl = wvl
    else:
        doe.change_wvl(wvl)
    light = doe.forward(light)

    if use_lens:
        lens = pado.optical_element.RefractiveLens(dim, param.DOE_pitch, param.focal_length, wvl, args.device)
        light = lens.forward(light.clone())

    aperture = Aperture(dim, param.DOE_pitch, param.aperture_diamter, param.aperture_shape, wvl, args.device)
    light = aperture.forward(light.clone())

    if full_psf:
        scale = (param.full_psf_scale[0], param.full_psf_scale[1])
    else:
        scale = (param.target_plane_scale[0], param.target_plane_scale[1])

    # Compute PSF where peak resides (similar to shifting PSF so that max value to be at center)
    if args.spatially_varying_PSF:
        ft = args.SVPSF_fitting_table
        shift_y, shift_x = ft[round(theta[0])+len(ft[-2])//2, round(theta[1])+len(ft[-1])//2]
        offset_y = -shift_y*param.camera_pitch
        offset_x = -shift_x*param.camera_pitch
        offset = (offset_y+offset[0], offset_x+offset[1])
    # debugging
    # light_prop = prop.forward(light, param.sensor_dist, offset=offset, variable_offset_indices=variable_offset_indices, linear=False, scale=scale)
    light_prop = prop.forward(light, param.sensor_dist, offset=offset, variable_offset_indices=variable_offset_indices, linear=False, scale=scale)
    # debugging
    psf = light_prop.get_intensity()

    # resize 
    if propagator!='RS':
        if args.resizing_method == 'original':
            psf = F.interpolate(psf, scale_factor=light_prop.pitch / param.DOE_pitch)
            psf = sample_psf(psf, param.DOE_sample_ratio)
        else:
            psf = F.interpolate(psf, scale_factor=light_prop.pitch/(param.camera_pitch* param.image_sample_ratio), 
                                mode=args.resizing_method)
    psf_size = psf.shape

    # debugging
    # print('psf size before padding: ', psf.shape)
    # debugging

    # Make psf dimensions of RGB to be same. 
    # In Fraunhofer's propagation, output field size depends on wavelength.
    if (pad is True) and (propagator=='Fraunhofer'):
        # compute dimensions of the Fraunhofer propagation result
        bw_r = light.get_bandwidth()[0]
        bw_c = light.get_bandwidth()[1]
        if bw_r != bw_c:
            raise NotImplementedError("Need to implement for the case when image shape is not square.")
        pitch_after_propagation = (np.array(param.broadband_wvls)) * np.array(param.sensor_dist / bw_r)
        scale_factors=pitch_after_propagation/(param.camera_pitch* param.image_sample_ratio)
        scale_factors_max = max(scale_factors)
        max_R = int(scale_factors_max * param.R) # round down
        wl, wr = compute_pad_size(psf_size[-1], max_R)
        hl, hr = compute_pad_size(psf_size[-2], max_R)
        psf = F.pad(psf, (wl, wr, hl, hr), "constant", 0)
        psf_size = psf.shape

    # debugging
    if propagator=='Fraunhofer':
        print('psf size after padding: ', psf.shape)
    # debugging

    cutoff = np.tan(np.arcsin(wvl/(2*param.DOE_pitch)))*param.focal_length / param.equiv_camera_pitch

    # debugging
    if propagator=='Fraunhofer':
        print('cutoff value: ', cutoff)
        DOE_mask = edge_mask(max(psf_size)//2, cutoff, args.device)
        psf = psf * DOE_mask 
    # debugging 
     
    if normalize:
        psf = psf / torch.sum(psf)

    return psf

def compute_psf_intensity_sum(wvl, depth, doe, args, use_lens=True, offset=(0, 0), target_plane_sample_trajectory=None, theta=(0, 0)):
    '''compute sum of intensities of the PSF. To remove light from some parts of the PSF plane.
    Args:
        depth: propagation distance
        use_lens: If True, DOE is placed in front of the lens.
        offset: (offset_y, offset_x) tuple. metric is meter. coordinate system follows matrix indexing ((0,0) is the top left corner.)
        theta: incident angle (theta_y, theta_x) of the point light source
    '''

    param = args.param
    prop = Propagator('SBL_ASM_intensity_sum')
    dim = (1,1,param.R, param.C)

    # compute compute (dy, dx) for set_spherical_light function
    dy = np.tan(np.deg2rad(theta[0])) * (param.focal_length + depth.numpy())
    dx = np.tan(np.deg2rad(theta[1])) * (param.focal_length + depth.numpy())
    
    # set incident point light source
    light = Light(dim, param.DOE_pitch, wvl, device=args.device)
    light.set_spherical_light(depth.numpy(), dx=dx, dy=dy)

    if args.constant_wvl_phase:
        doe.wvl = wvl
    else:
        doe.change_wvl(wvl)
    light = doe.forward(light)

    if use_lens:
        lens = pado.optical_element.RefractiveLens(dim, param.DOE_pitch, param.focal_length, wvl, args.device)
        light = lens.forward(light.clone())

    aperture = Aperture(dim, param.DOE_pitch, param.aperture_diamter, param.aperture_shape, wvl, args.device)
    light = aperture.forward(light.clone())

    # Compute PSF where peak resides (similar to shifting PSF so that max value to be at center)
    if args.spatially_varying_PSF:
        ft = args.SVPSF_fitting_table
        shift_y, shift_x = ft[round(theta[0])+len(ft[-2])//2, round(theta[1])+len(ft[-1])//2]
        offset_y = -shift_y*param.camera_pitch
        offset_x = -shift_x*param.camera_pitch
        offset = (offset_y, offset_x)

    intensity_sum = prop.forward(light, param.sensor_dist, offset=offset, target_plane_sample_trajectory=target_plane_sample_trajectory)
    return intensity_sum


def plot_depth_based_psf(doe, args, depths, wvls = 'RGB', merge_channel = False, pad=True, use_lens=None, propagator=None, eval=False, offset=(0, 0), theta=(0, 0), normalize=True):
    param = args.param
    psfs = []
    if use_lens is None:
        use_lens = args.use_lens 
    if propagator is None:
        propagator = args.propagator
    if wvls == 'RGB':
        for i in range(len(param.wvls)):
            wvl = param.wvls[i]
            psf_depth = []
            for z in depths:
                psf = compute_psf_arbitrary_prop(wvl, 
                                                torch.tensor(z) if not isinstance(z, torch.Tensor) else z.clone().detach(), 
                                                doe, 
                                                args, 
                                                propagator=propagator,
                                                use_lens=use_lens,
                                                offset=offset,
                                                theta=theta,
                                                pad=pad,
                                                normalize=normalize)
                psf_depth.append(psf.detach() if eval else psf)
            psfs.append(torch.cat(psf_depth, -1))
        if merge_channel:
            psfs = torch.cat(psfs, 1)
        else:
            psfs = torch.cat(psfs, -2)
    elif isinstance(wvls, list) and len(wvls) == 3:
        for wvl in wvls:
            psf_depth = []
            for z in depths:
                psf = compute_psf_arbitrary_prop(wvl, 
                                                torch.tensor(z) if not isinstance(z, torch.Tensor) else z.clone().detach(), 
                                                doe, 
                                                args, 
                                                propagator=propagator,
                                                use_lens=use_lens,
                                                offset=offset,
                                                theta=theta,
                                                pad=pad,
                                                normalize=normalize)
                psf_depth.append(psf.detach() if eval else psf)
            psfs.append(torch.cat(psf_depth, -1))
        if merge_channel:
            psfs = torch.cat(psfs, 1)
        else:
            psfs = torch.cat(psfs, -2)
    else:
        psf_depth = []
        for z in depths:
            # debugging
            # print('@@@@@@@wvls: ', wvls)
            # debugging
            psf = compute_psf_arbitrary_prop(wvls, 
                                            torch.tensor(z) if not isinstance(z, torch.Tensor) else z.clone().detach(), 
                                            doe, 
                                            args, 
                                            propagator=propagator,
                                            use_lens=use_lens,
                                            offset=offset,
                                            theta=theta,
                                            pad=pad,
                                            normalize=normalize)
            psf_depth.append(psf.detach() if eval else psf)
        psfs = torch.cat(psf_depth, -1)

    log_psfs = torch.log(psfs+1e-9)
    log_psfs = log_psfs - torch.min(log_psfs)
    log_psfs = log_psfs / torch.max(log_psfs)
    return psfs, log_psfs

def plot_psf_array(psfs, param, center_R=10, gap=10, g=1.5, log_scale=False):
    psfs_singles = torch.split(psfs, param.img_res, dim=-1)
    cnt = len(psfs_singles)
    canvas = np.ones([param.img_res, param.img_res * cnt + gap * (cnt - 1), 3])
    for i in range(cnt):
        psf = psfs_singles[i][0, 0].cpu().numpy() if psfs.shape[1] == 1 else psfs_singles[i][0].permute(1, 2, 0).cpu().numpy()
        canvas[:, i * (param.img_res + gap):i * (param.img_res + gap) + param.img_res, :] = viz_psf(psf, param, center_R=center_R if i < 3 else None, g=g, log_scale=log_scale)

    plt.figure(figsize=(30, 10))
    plt.imshow(canvas)
    plt.axis('off')  # Hide axes
    plt.show()
    return canvas

def viz_psf(psf, param, center_R=None, g=2.2, weight=4, log_scale=False):
    if log_scale:
        # Apply log scale; add a small constant to avoid log(0)
        psf = np.log(psf + 1e-10)
        psf = (psf - np.min(psf)) / (np.max(psf) - np.min(psf))  # Normalize after log
    else:
        psf = (psf / np.max(psf)) ** (1/g)

    if center_R is not None:
        size = 2 * center_R
        w = h = int(param.img_res / 2 - center_R)
        psf_center = psf[h:h + size, w:w + size]
        psf_center = cv2.resize(psf_center, (center_R * 15, center_R * 15), interpolation=cv2.INTER_NEAREST)
        psf[:center_R * 15, :center_R * 15] = psf_center

        # Use a colormap if the PSF is single-channel
        if len(psf.shape) == 2:
            psf = plt.cm.hot(psf)[:, :, :-1]

        # Highlight the center region and add borders for clarity
        psf[:center_R * 15 + weight, center_R * 15: center_R * 15 + weight, :] = 1
        psf[center_R * 15: center_R * 15 + weight, :center_R * 15 + weight, :] = 1
    else:
        if len(psf.shape) == 2:
            psf = plt.cm.hot(psf)[:, :, :-1]
    return psf

def create_radial_gaussian_psf(batch_size, num_channels, dim, std_dev, device='cpu'):
    """
    Create a radial Gaussian PSF with the center having the lowest intensity,
    increasing towards the edges, with dimensions suitable for batch processing.

    Args:
    - batch_size (int): The number of PSFs in a batch.
    - num_channels (int): The number of channels in the PSF.
    - dim (int): The height and width of the PSF array (assumed square).
    - std_dev (float): The standard deviation of the Gaussian distribution.
    - device (str): The computation device ('cpu' or 'cuda').

    Returns:
    - torch.Tensor: The generated PSF tensor of shape (batch_size, num_channels, dim, dim).
    """
    # Create a grid of (x, y) coordinates
    x = torch.linspace(-dim/2, dim/2, steps=dim, device=device)
    y = torch.linspace(-dim/2, dim/2, steps=dim, device=device)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')

    # Calculate the squared distance from the center
    sq_distances = x_grid ** 2 + y_grid ** 2

    # Convert squared distances to a Gaussian distribution
    gaussian_psf = torch.exp(-sq_distances / (2 * std_dev ** 2))

    # Normalize PSF to have a minimum value of 0 at the center and max at edges
    gaussian_psf = 1 - gaussian_psf / torch.max(gaussian_psf)

    # Expand to full dimensions (batch_size, num_channels, R, C)
    gaussian_psf = gaussian_psf.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    gaussian_psf = gaussian_psf.expand(batch_size, num_channels, dim, dim)

    return gaussian_psf

# Parameter and Configuration Management

def convert_resolution(param, args):
    # dataset
    if args.obstruction == 'fence':
        param.dataset_dir = args.fence_dataset_dir
        param.data_resolution = [512,768]
    elif args.obstruction == 'raindrop' or 'dirt' or 'dirt_raindrop':
        param.training_dir = args.dirt_raindrop_dataset_train_dir
        param.val_dir = args.dirt_raindrop_dataset_val_dir
        param.data_resolution = [1024, 2048]
    else:
        assert False, "undefined obstruction"

    # convert resolution and pitch size
    param.equiv_image_size = param.img_res * param.image_sample_ratio # image resolution before downsampling in camera pixel pitch
    param.equiv_crop_size = int(param.equiv_image_size * param.camera_pitch / param.background_pitch)  # convert to background pixel pitch 
    return param

def save_settings(args, param):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    else:
        raise Exception("The directory already exists!") 
    args_dict = vars(args)
    with open(os.path.join(args.result_path,'args.json'), "w") as f:
        json.dump(args_dict, f, indent=4, sort_keys=False)
    shutil.copy(args.param_file, args.result_path)
    if args.pretrained_DOE is not None:
        shutil.copy(args.pretrained_DOE, os.path.join(args.result_path, 'init'))
    if args.pretrained_network is not None:
        shutil.copy(args.pretrained_network, os.path.join(args.result_path, 'init'))
    args.param = param

def last_save(ckpt_path, file_format):
    return sorted(glob(os.path.join(ckpt_path, file_format)))[-1]

# Image Processing and Fourier Transforms

def fft_convolve2d(image, kernel, linear=True):
    # Ensure the kernel is centered
    b, c, ih, iw = image.shape

    if kernel.dim() == 2:  # If the kernel has shape [W, H]
        kernel = kernel.unsqueeze(0).unsqueeze(0).expand(b, c, -1, -1)
    elif kernel.dim() == 3:  # If the kernel has shape [C, W, H]
        kernel = kernel.unsqueeze(0).expand(b, -1, -1, -1)

    # # crop only meaningful area. We don't need area exceeds 2X size of the image
    # if kernel.shape[-2] > ih*2 and kernel.shape[-1] > iw*2: 
    #     kernel = kernel.clone()
    #     kernel = kernel[..., kernel.shape[-2]//2-ih:kernel.shape[-2]//2+ih, kernel.shape[-1]//2-iw:kernel.shape[-1]//2+iw]
    # elif kernel.shape[-2] > ih*2: 
    #     kernel = kernel.clone()
    #     kernel = kernel[..., kernel.shape[-2]//2-ih:kernel.shape[-2]//2+ih, :]
    # elif kernel.shape[-1] > iw*2: 
    #     kernel = kernel.clone()
    #     kernel = kernel[..., :, kernel.shape[-1]//2-iw:kernel.shape[-1]//2+iw]
    _, _, kh, kw = kernel.shape

    # Compute the size of the convolution result. This includes zero padding to avoid circular convolution and ensure linear convolution
    if linear:
        conv_shape_image = (b, c, ih + kh - 1, iw + kw - 1)
        conv_shape_kernel = (b, kernel.shape[-3], ih + kh - 1, iw + kw - 1)
    else:
        raise NotImplementedError('circular convolution is not implemented in fft_convolved2d function!!!')
        
    # Pad the image
    padded_image = torch.zeros(conv_shape_image, dtype=image.dtype, device=image.device)
    padded_image[:, :, :ih, :iw] = image
    
    # Pad the kernel
    padded_kernel = torch.zeros(conv_shape_kernel, dtype=kernel.dtype, device=kernel.device)
    padded_kernel[:, :, :kh, :kw] = kernel
    # Center the kernel (PSF)
    padded_kernel = torch.roll(padded_kernel, shifts=(-kh//2, -kw//2), dims=(2, 3))
    
    image_fft = fft(padded_image, shift=False)
    kernel_fft = fft(padded_kernel, shift=False)
    convolved = ifft(image_fft*kernel_fft, shift=False)
    
    # Extract the original image size
    convolved = convolved[:, :, :ih, :iw].real
    return convolved

def fft_convolve2d_image_reflection(image, kernel):
    # Use this function to avoid zero-padded linear convolution making image darker.

    # Ensure the kernel is centered
    b, c, ih, iw = image.shape

    if kernel.dim() == 2:  # If the kernel has shape [W, H]
        kernel = kernel.unsqueeze(0).unsqueeze(0).expand(b, c, -1, -1)
    elif kernel.dim() == 3:  # If the kernel has shape [C, W, H]
        kernel = kernel.unsqueeze(0).expand(b, -1, -1, -1)

    _, _, kh, kw = kernel.shape

    # Compute the size of the convolution result. 
    # This includes zero padding to avoid circular convolution and ensure linear convolution.
    # Note that we only zero padd the PSF kernel, and will do the reflection padding for the input image.
    conv_shape_kernel = (b, kernel.shape[-3], ih + kh - 1, iw + kw - 1)
        
    # Pad the image
    reflection_operation = nn.ReplicationPad2d((0, kh-1, 0, kw-1))
    padded_image = reflection_operation(image)
    
    # Pad the kernel (PSF)
    padded_kernel = torch.zeros(conv_shape_kernel, dtype=kernel.dtype, device=kernel.device)
    padded_kernel[:, :, :kh, :kw] = kernel
    # Center the kernel (PSF)
    padded_kernel = torch.roll(padded_kernel, shifts=(-kh//2, -kw//2), dims=(2, 3))
    
    image_fft = fft(padded_image, shift=False)
    kernel_fft = fft(padded_kernel, shift=False)
    convolved = ifft(image_fft*kernel_fft, shift=False)
    
    # Extract the original image size
    convolved = convolved[:, :, :ih, :iw].real
    return convolved

# Image Formation and Simulation
def image_formation(image_far, DOE, compute_obstruction, args, channel_idx=None, z_near = None, wvls='RGB', theta=(0, 0), pad=True, far_disparity=False, concat=False):
    param = args.param
    if z_near is None:
        z_near = randuni(param.depth_near_min, param.depth_near_max, 1)[0] # randomly sample the near-point depth from a range

    z_far = randuni(param.depth_far_min, param.depth_far_max, 1)[0] # randomly sample the far-point depth from a range
    image_near, mask = compute_obstruction(image_far.get_intensity() if isinstance(image_far, pado.light.Light) 
                                           else image_far, z_near, args)
    
    if mask.shape[3] > 1:
        mask = mask[:,0:1,...]
    
    if not args.train_RGB:
        image_near = torch.mean(image_near, dim=-3, keepdim=True)

    image_far = image_far.to(torch.float32)
    image_near = image_near.to(torch.float32)
    mask = mask.to(torch.float32)

    psf_far, _ = plot_depth_based_psf(DOE, args, depths = [z_far], wvls=wvls, merge_channel = True, pad=pad, theta=theta, normalize=False)
    psf_near, _ = plot_depth_based_psf(DOE, args, depths = [z_near], wvls=wvls, merge_channel = True, pad=pad, theta=theta, normalize=False)

    if args.normalizing_method == 'new' and (psf_far.shape[-2] > param.camera_resolution[-2] and psf_far.shape[-1] > param.camera_resolution[-1]):
        normalizer = psf_far[...,psf_far.shape[-2]//2-param.camera_resolution[-2]//2:psf_far.shape[-2]//2+param.camera_resolution[-2]//2,
                            psf_far.shape[-1]//2-param.camera_resolution[-1]//2:psf_far.shape[-1]//2+param.camera_resolution[-1]//2].sum()
    else:
        normalizer = psf_far.sum() if psf_far.shape[-3]==1 else psf_far[0,1,...].sum()

    # compute intensity sum of the PSF for the area that sensor respond to
    if (psf_far.shape[-2] > param.camera_resolution[-2] and psf_far.shape[-1] > param.camera_resolution[-1]):
        psf_sensor_sum = psf_far[...,psf_far.shape[-2]//2-param.camera_resolution[-2]//2:psf_far.shape[-2]//2+param.camera_resolution[-2]//2,
                            psf_far.shape[-1]//2-param.camera_resolution[-1]//2:psf_far.shape[-1]//2+param.camera_resolution[-1]//2].sum()
    else:
        psf_sensor_sum =  psf_far.sum() if psf_far.shape[-3]==1 else psf_far[0,1,...].sum()

    if args.propagator == 'Fraunhofer':
        psf_far = psf_far / psf_far.sum() 
        psf_near = psf_near / psf_near.sum() 
    else:
        psf_far = psf_far/normalizer
        psf_near = psf_near/normalizer

    if channel_idx is not None:
        image_far = image_far[:, channel_idx:channel_idx+1, ...]
        image_near = image_near[:, channel_idx:channel_idx+1, ...]

    # Conserve total light intensity
    # convolved_far = fft_convolve2d(image_far.get_intensity() if isinstance(image_far, pado.light.Light) else image_far, 
    #                                psf_far)
    convolved_far = fft_convolve2d_image_reflection(image_far.get_intensity() if isinstance(image_far, pado.light.Light) else image_far, 
                                   psf_far)

    # Adjust brightness
    scale_factor_brightness = torch.clamp(image_far.mean() / convolved_far.mean(), max=args.brightness_clamp)
    print('brightness scale factor:', scale_factor_brightness)
    psf_far = psf_far * scale_factor_brightness
    psf_near = psf_near * scale_factor_brightness
    convolved_far = convolved_far * scale_factor_brightness
    
    if param.metasurface_array is not None:
        if concat:
            img_convs = list()
        else:
            img_convs = torch.zeros_like(image_near)
        masks = torch.zeros_like(mask)
        disparity = round((param.focal_length/param.camera_pitch) * (param.baseline) / z_near.item())
        disparities_y, disparities_x = np.meshgrid(np.linspace(-(param.metasurface_array[0]-1)/2*disparity, (param.metasurface_array[0]-1)/2*disparity, param.metasurface_array[0]), 
                                            np.linspace(-(param.metasurface_array[1]-1)/2*disparity, (param.metasurface_array[1]-1)/2*disparity, param.metasurface_array[1]), indexing='ij')
        for disparity_y_, disparity_x_ in zip(disparities_y.ravel(), disparities_x.ravel()):
            disparity_y_ = round(disparity_y_.item())
            disparity_x_ = round(disparity_x_.item())
            convolved_near = shift_image(fft_convolve2d(image_near, psf_near), disparity_y_, disparity_x_)
            convolved_mask = shift_image(fft_convolve2d(mask, psf_near), disparity_y_, disparity_x_)
            convolved_mask = torch.clamp(1.5 * convolved_mask, 0, 1)

            if far_disparity:
                img_conv = shift_image(convolved_far, round(disparity_y_/(z_far.item()/z_near.item())), 
                                   round(disparity_x_/(z_far.item()/z_near.item()))) * (1 - convolved_mask) + convolved_near * convolved_mask
            else:
                img_conv = convolved_far * (1 - convolved_mask) + convolved_near * convolved_mask
            
            if concat:
                img_convs.append(img_conv)
            else:
                img_convs = img_convs + img_conv
            masks = masks + shift_image(mask, disparity_y_, disparity_x_)
        if concat:
            img_conv = torch.cat(img_convs, dim=-3)
        else:
            img_conv = img_convs / len(disparities_y.ravel())
        mask = torch.clamp(masks, 0, 1)
    else:
        convolved_near = fft_convolve2d(image_near, psf_near)
        convolved_mask = fft_convolve2d(mask, psf_near)
        convolved_mask = torch.clamp(1.5 * convolved_mask, 0, 1)
        img_conv = convolved_far * (1 - convolved_mask) + convolved_near * convolved_mask

    if far_disparity:
        H, W = image_far.shape[2:]
        start_x = W//2-param.cropped_img_res//2
        end_x = W//2+param.cropped_img_res//2
        start_y = H//2-param.cropped_img_res//2
        end_y = H//2+param.cropped_img_res//2
        image_far = image_far.clone()[..., start_y:end_y, start_x:end_x]
        image_near = image_near.clone()[..., start_y:end_y, start_x:end_x]
        mask = mask.clone()[..., start_y:end_y, start_x:end_x]
        convolved_far = convolved_far.clone()[..., start_y:end_y, start_x:end_x]
        convolved_near = convolved_near.clone()[..., start_y:end_y, start_x:end_x]
        convolved_mask = convolved_mask.clone()[..., start_y:end_y, start_x:end_x]
        img_conv = img_conv.clone()[..., start_y:end_y, start_x:end_x]

    return image_far, image_near, mask, convolved_far, convolved_near, convolved_mask, img_conv, psf_near, psf_far, psf_sensor_sum

#--------- Single sensor synthetic aperture image formation ---------#
def image_formation_single_sensor_synthetic_aperture(image_far, DOE, compute_obstruction, args, channel_idx=None, z_near = None, wvls='RGB', theta=(0, 0), pad=True):
    param = args.param
    if z_near is None:
        z_near = randuni(param.depth_near_min, param.depth_near_max, 1)[0] # randomly sample the near-point depth from a range

    z_far = randuni(param.depth_far_min, param.depth_far_max, 1)[0] # randomly sample the far-point depth from a range
    image_near, mask = compute_obstruction(image_far.get_intensity() if isinstance(image_far, pado.light.Light) 
                                           else image_far, z_near, args)
    
    if mask.shape[3] > 1:
        mask = mask[:,0:1,...]
    
    if not args.train_RGB:
        image_near = torch.mean(image_near, dim=-3, keepdim=True)

    image_far = image_far.to(torch.float32)
    image_near = image_near.to(torch.float32)
    mask = mask.to(torch.float32)

    # propagation offset 
    propagation_offset = (param.baseline/2, param.baseline/2)

    # light source offset
    light_source_offset = (param.baseline/2, param.baseline/2)

    psf_far = plot_depth_based_psf_single_sensor_synthetic_aperture(DOE, args, depths = [z_far], wvls=wvls, merge_channel=True, 
                                                                       light_source_offset=light_source_offset, 
                                                                       propagation_offset=propagation_offset, 
                                                                       normalize=False)
    psf_near = plot_depth_based_psf_single_sensor_synthetic_aperture(DOE, args, depths = [z_near], wvls=wvls, merge_channel=True, 
                                                                        light_source_offset=light_source_offset, 
                                                                        propagation_offset=propagation_offset, 
                                                                        normalize=False)
    
    if args.normalizing_method == 'new' and (psf_far.shape[-2] > param.camera_resolution[-2] and psf_far.shape[-1] > param.camera_resolution[-1]):
        normalizer = psf_far[...,psf_far.shape[-2]//2-param.camera_resolution[-2]//2:psf_far.shape[-2]//2+param.camera_resolution[-2]//2,
                            psf_far.shape[-1]//2-param.camera_resolution[-1]//2:psf_far.shape[-1]//2+param.camera_resolution[-1]//2].sum()
    else:
        normalizer = psf_far.sum() if psf_far.shape[-3]==1 else psf_far[0,1,...].sum()

    psf_far = psf_far/normalizer
    psf_near = psf_near/normalizer

    if channel_idx is not None:
        image_far = image_far[:, channel_idx:channel_idx+1, ...]
        image_near = image_near[:, channel_idx:channel_idx+1, ...]

    convolved_far = fft_convolve2d(image_far.get_intensity() if isinstance(image_far, pado.light.Light) else image_far, 
                                   psf_far)
    
    # Adjust brightness
    scale_factor_brightness = torch.clamp(image_far.mean() / convolved_far.mean(), max=args.brightness_clamp)
    print('brightness scale factor:', scale_factor_brightness)
    psf_far = psf_far * scale_factor_brightness
    psf_near = psf_near * scale_factor_brightness
    convolved_far = convolved_far * scale_factor_brightness
    
    convolved_near = fft_convolve2d(image_near, psf_near)
    convolved_mask = fft_convolve2d(mask, psf_near)
    convolved_mask = torch.clamp(1.5 * convolved_mask, 0, 1)
    img_conv = convolved_far * (1 - convolved_mask) + convolved_near * convolved_mask

    return image_far, image_near, mask, convolved_far, convolved_near, convolved_mask, img_conv, psf_near, psf_far, normalizer

def plot_depth_based_psf_single_sensor_synthetic_aperture(doe, args, depths, wvls = 'RGB', merge_channel=False, use_lens=None, propagator=None, light_source_offset=(0, 0), propagation_offset=(0, 0), normalize=True):
    param = args.param
    psfs = []
    if use_lens is None:
        use_lens = args.use_lens 
    if propagator is None:
        propagator = args.propagator
    if wvls == 'RGB':
        for i in range(len(param.wvls)):
            wvl = param.wvls[i]
            psf_depth = []
            for z in depths:
                psf = compute_psf_single_sensor_synthetic_aperture(wvl, 
                                            torch.tensor(z) if not isinstance(z, torch.Tensor) else z.clone().detach(), 
                                            doe, 
                                            args, 
                                            propagator=propagator,
                                            use_lens=use_lens,
                                            light_source_offset=light_source_offset, 
                                            propagation_offset=propagation_offset,
                                            normalize=normalize)
                psf_depth.append(psf)
            psfs.append(torch.cat(psf_depth, -1))
        if merge_channel:
            psfs = torch.cat(psfs, 1)
        else:
            psfs = torch.cat(psfs, -2)

    else:
        psf_depth = []
        for z in depths:
            psf = compute_psf_single_sensor_synthetic_aperture(wvls, 
                                            torch.tensor(z) if not isinstance(z, torch.Tensor) else z.clone().detach(), 
                                            doe, 
                                            args, 
                                            propagator=propagator,
                                            use_lens=use_lens,
                                            light_source_offset=light_source_offset, 
                                            propagation_offset=propagation_offset,
                                            normalize=normalize)
            psf_depth.append(psf)
        psfs = torch.cat(psf_depth, -1)

    return psfs

def compute_psf_single_sensor_synthetic_aperture(wvl, depth, doe, args, propagator='Fraunhofer', use_lens=True, light_source_offset=(0, 0), propagation_offset=(0, 0), full_psf=False, normalize=True):

    param = args.param
    prop = Propagator(propagator)
    dim = (1,1,param.R, param.C)

    dx, dy = light_source_offset
    
    # set incident point light source
    light = Light(dim, param.DOE_pitch, wvl, device=args.device)
    # plt.imsave(os.path.join(args.result_path, 'light', f'original_light_depth{depth.item()}.png'), np.clip(light.get_phase().cpu().detach().numpy()[0, 0, :, :], 0, 1))
    light.set_spherical_light(depth.numpy(), dx=dx, dy=dy)
    # plt.imsave(os.path.join(args.result_path, 'light', f'spherical_light_depth{depth.item()}.png'), np.clip(light.get_phase().cpu().detach().numpy()[0, 0, :, :], 0, 1))
    # print('propagation offset: ', propagation_offset, 'light source offset: ', light_source_offset)

    if args.constant_wvl_phase:
        doe.wvl = wvl
    else:
        doe.change_wvl(wvl)
    light = doe.forward(light)
    # plt.imsave(os.path.join(args.result_path, 'light', f'doe_passed_{depth.item()}.png'), np.clip(light.get_phase().cpu().detach().numpy()[0, 0, :, :], 0, 1))

    if use_lens:
        lens = pado.optical_element.RefractiveLens(dim, param.DOE_pitch, param.focal_length, wvl, args.device)
        light = lens.forward(light.clone())

    aperture = Aperture(dim, param.DOE_pitch, param.aperture_diamter, param.aperture_shape, wvl, args.device)
    light = aperture.forward(light.clone())

    if full_psf:
        scale = (param.full_psf_scale[0], param.full_psf_scale[1])
    else:
        scale = (param.target_plane_scale[0], param.target_plane_scale[1])


    light_prop = prop.forward(light, param.sensor_dist, offset=propagation_offset, variable_offset_indices=None, scale=scale)
    summed_field = light_prop.get_field() + torch.flip(light_prop.get_field(), dims=[-1]) + torch.flip(light_prop.get_field(), dims=[-2]) + torch.flip(light_prop.get_field(), dims=[-1, -2])
    light_prop.set_field(summed_field)
    psf = light_prop.get_intensity()

    # plt.imsave(os.path.join(args.result_path, 'psf', f'PSF_propagated_{depth.item()}.png'), np.clip(psf.cpu().detach().numpy()[0, 0, :, :], 0, 1))

    # resize 
    if propagator!='RS':
        if args.resizing_method == 'original':
            psf = F.interpolate(psf, scale_factor=light_prop.pitch / param.DOE_pitch)
            psf = sample_psf(psf, param.DOE_sample_ratio)
        else:
            psf = F.interpolate(psf, scale_factor=light_prop.pitch/(param.camera_pitch* param.image_sample_ratio), 
                                mode=args.resizing_method)

    if normalize:
        psf = psf / torch.sum(psf)

    return psf

def evaluate_DOE_single_sensor_synthetic_aperture(DOE_phase, total_step, args, writer=None, net=None, save_images=True, RGB=False, obstruction_reinforcement=False, phase_noise=False, image_noise=False, vis_psf=False, simple_psf=False, change_wvl=False, adjust=False):
    # Evaluate DOE using predetermined mask and obstacle image
    param = args.param
    
    if phase_noise and args.phase_noise_stddev is not None:
        DOE_phase = DOE_phase + torch.randn((DOE_phase.shape), device=args.device) * args.phase_noise_stddev

    image_near = torch.tensor(np.load(os.path.join(args.eval_path, 'img_near_'+str(param.img_res)+'_0022.npy')), device=args.device).detach()
    image_far = torch.tensor(np.load(os.path.join(args.eval_path, 'img_far_'+str(param.img_res)+'_0022.npy')), device=args.device).detach()
    mask = torch.tensor(np.load(os.path.join(args.eval_path, 'mask_'+str(param.img_res)+'_0022.npy')), device=args.device).detach()

    DOE_train = DOE((1,1,param.R, param.R), param.DOE_pitch, param.material, wvl=param.DOE_wvl, device=args.device)
    DOE_train.set_phase_change(DOE_phase.detach())  
    psf_far = plot_depth_based_psf_single_sensor_synthetic_aperture(DOE_train, args, [(param.depth_far_min+param.depth_far_max)/2], 
                                                                    wvls='RGB' if RGB is True else param.DOE_wvl, merge_channel = True, 
                                                                    propagation_offset = (param.baseline/2, param.baseline/2), 
                                                                    light_source_offset = (param.baseline/2, param.baseline/2),
                                                                    normalize=False)
    psf_near = plot_depth_based_psf_single_sensor_synthetic_aperture(DOE_train, args, [(param.depth_near_min+param.depth_near_max)/2], 
                                                                    wvls='RGB' if RGB is True else param.DOE_wvl, merge_channel = True, 
                                                                    propagation_offset = (param.baseline/2, param.baseline/2), 
                                                                    light_source_offset = (param.baseline/2, param.baseline/2),
                                                                    normalize=False)

    if args.normalizing_method == 'new' and (psf_far.shape[-2] > param.camera_resolution[-2] and psf_far.shape[-1] > param.camera_resolution[-1]): # approximate
        normalizer = psf_far[:, 1 if RGB else 0, psf_far.shape[-2]//2-param.camera_resolution[-2]//2:psf_far.shape[-2]//2+param.camera_resolution[-2]//2,
                                psf_far.shape[-1]//2-param.camera_resolution[-1]//2:psf_far.shape[-1]//2+param.camera_resolution[-1]//2].sum()
    else:
        normalizer = psf_far[:,1,...].sum() if RGB else psf_far.sum()

    psf_near = psf_near / normalizer
    psf_far = psf_far / normalizer

    convolved_far = fft_convolve2d(image_far.get_intensity() if isinstance(image_far, pado.light.Light) else image_far, 
                                psf_far)

    # Adjust brightness
    scale_factor_brightness = torch.clamp(image_far.mean() / convolved_far.mean(), max=args.brightness_clamp)
    print('brightness scale factor:', scale_factor_brightness)
    psf_far = psf_far * scale_factor_brightness
    psf_near = psf_near * scale_factor_brightness
    convolved_far = convolved_far * scale_factor_brightness

    log_psf_near = torch.log(psf_near + 1e-9)
    log_psf_near -= torch.min(log_psf_near)
    log_psf_near /= torch.max(log_psf_near)

    log_psf_far = torch.log(psf_far + 1e-9)
    log_psf_far -= torch.min(log_psf_far)
    log_psf_far /= torch.max(log_psf_far)

    if vis_psf:
        if psf_near.shape[1]==1: 
            colors = ['grayscale']
        else:
            colors = ['Red', 'Green', 'Blue']
        for c, color in enumerate(colors):
            plt.figure(figsize=(10, 10))
            plt.title(f'PSF near {color}')
            plt.imshow(psf_near[0, c, ...].cpu().numpy(), vmax=0.0001) #  vmax=0.00001
            plt.colorbar()
            plt.figure(figsize=(10, 10))
            plt.title(f'PSF far {color}')
            plt.imshow(psf_far[0, c, ...].cpu().numpy(), vmax=0.001)
            plt.colorbar()

            plt.figure(figsize=(10, 10))
            plt.title(f'PSF near log {color}')
            plt.imshow(log_psf_near[0, c, ...].cpu().numpy())
            plt.colorbar()
            plt.figure(figsize=(10, 10))
            plt.title(f'PSF far log{color}')
            plt.imshow(log_psf_far[0, c, ...].cpu().numpy())
            plt.colorbar()

            plt.figure(figsize=(5, 5))
            plt.imshow(psf_near[0, c, ...][psf_near.shape[2]//2-30:psf_near.shape[2]//2+30, psf_near.shape[3]//2-30:psf_near.shape[3]//2+30].cpu().numpy())
            plt.colorbar()
            plt.title(f'PSF_near_center {color}')
            plt.figure(figsize=(5, 5))
            plt.imshow(psf_far[0, c, ...][psf_far.shape[2]//2-30:psf_far.shape[2]//2+30, psf_far.shape[3]//2-30:psf_far.shape[3]//2+30].cpu().numpy())
            plt.colorbar()
            plt.title(f'PSF_far_center {color}')

            airy_disk_diameter = round(2.44*param.wvls[c]*(param.focal_length/(param.DOE_pitch*param.R))/param.camera_pitch)
            ylim=psf_far.max().cpu().numpy()*airy_disk_diameter*3

            plt.figure(figsize=(13, 5))
            graph_1d = psf_far[0, c, ...].cpu().numpy()
            plt.plot(graph_1d.sum(axis=0))
            plt.ylim([0, ylim])
            plt.title(f"""PSF far {color} 1D visualization, 
peak sum={graph_1d.sum(axis=0)[psf_far.shape[-1]//2-airy_disk_diameter:psf_far.shape[-1]//2+airy_disk_diameter].sum():.5f},
peak max={graph_1d.sum(axis=0).max():.5f}""")
            plt.show()

            plt.figure(figsize=(13, 5))
            graph_1d = psf_near[0, c, ...].cpu().numpy()
            plt.plot(graph_1d.sum(axis=0))
            plt.ylim([0, ylim])
            plt.title(f"""PSF near {color} 1D visualization, 
peak sum={graph_1d.sum(axis=0)[psf_far.shape[-1]//2-airy_disk_diameter:psf_far.shape[-1]//2+airy_disk_diameter].sum():.5f},
peak max={graph_1d.sum(axis=0).max():.5f}""")
            plt.show()

            ylim=psf_far.max().cpu().numpy()*airy_disk_diameter*2

            plt.figure(figsize=(13, 5))
            graph_1d = psf_far[0, c, psf_far.shape[-2]//2-airy_disk_diameter:psf_far.shape[-2]//2+airy_disk_diameter, :].cpu().numpy()
            plt.plot(graph_1d.sum(axis=0))
            plt.ylim([0, ylim])
            plt.title(f"""PSF far {color} center 1D visualization, 
peak sum={graph_1d.sum(axis=0)[psf_far.shape[-1]//2-airy_disk_diameter:psf_far.shape[-1]//2+airy_disk_diameter].sum():.5f},
peak max={graph_1d.sum(axis=0).max():.5f}""")
            plt.show()

            plt.figure(figsize=(13, 5))
            graph_1d = psf_near[0, c, psf_far.shape[-2]//2-airy_disk_diameter:psf_far.shape[-2]//2+airy_disk_diameter, :].cpu().numpy()
            plt.plot(graph_1d.sum(axis=0))
            plt.ylim([0, ylim])
            plt.title(f"""PSF near {color} center 1D visualization, 
peak sum={graph_1d.sum(axis=0)[psf_far.shape[-1]//2-airy_disk_diameter:psf_far.shape[-1]//2+airy_disk_diameter].sum():.5f},
peak max={graph_1d.sum(axis=0).max():.5f}""")
            plt.show()

            print(f'channel {c}, psf near center sum: {psf_near[0, c, ...][psf_near.shape[2]//2-5:psf_near.shape[2]//2+5, psf_near.shape[3]//2-5:psf_near.shape[3]//2+5].sum()}', 
                  f'channel {c}, psf far center sum: {psf_far[0, c, ...][psf_far.shape[2]//2-5:psf_far.shape[2]//2+5, psf_far.shape[3]//2-5:psf_far.shape[3]//2+5].sum()}')
  
    convolved_near = fft_convolve2d(image_near, psf_near)
    convolved_mask = fft_convolve2d(mask, psf_near)
    convolved_mask = torch.clamp(1.5 * convolved_mask, 0, 1)
    img_conv = convolved_far * (1 - convolved_mask) + convolved_near * convolved_mask

    if args.use_network:
        img_conv_before_net = img_conv.clone()
        img_conv = net(img_conv.to(torch.float32))
    
    # Adjust contrast
    scale_factor_contrast = torch.clamp(image_far.std()/img_conv.std(), max=args.contrast_clamp).item()
    img_conv = img_conv.mean() + scale_factor_contrast * (img_conv - img_conv.mean())

    if adjust:
        img_conv = adjust_brightness_and_contrast(img_conv, image_far)

    if image_noise and args.image_noise_stddev is not None:
        noise = torch.randn((img_conv.shape), device=args.device) * args.image_noise_stddev
        img_conv = torch.clamp(img_conv + noise.type_as(img_conv), 0, 1)
        convolved_far = torch.clamp(convolved_far + noise.type_as(img_conv), 0, 1)
    
    psnr = skimage.metrics.peak_signal_noise_ratio(
        image_far[0].permute(1, 2, 0).cpu().detach().numpy(), img_conv[0].permute(1, 2, 0).cpu().detach().numpy())
    l1_loss = args.l1_criterion(img_conv, image_far).detach().item()

    print('l1_loss: ', l1_loss, 'psnr: ', psnr, 'brightness multiplier: ', scale_factor_brightness)

    if writer is not None:       
        writer.add_scalar('Loss/train', l1_loss, total_step)  # Log loss for TensorBoard
        writer.add_scalar('PSNR/train', psnr, total_step)
        if args.use_da_loss:
            from models.DA_loss_functions import DA_loss
            da_loss = DA_loss(img_conv, image_far, args).detach().item()
            writer.add_scalar('DA_loss/train', da_loss, total_step)
        writer.add_image('image near', image_near[0].cpu().detach(), total_step)
        writer.add_image('image_far', image_far[0].cpu().detach(), total_step)
        writer.add_image('convolved far only', convolved_far[0].cpu().detach(), total_step)
        writer.add_image('convolved image', img_conv[0].cpu().detach(), total_step)
        writer.add_image('psf near (1st channel)', psf_near.cpu().detach()[0, 0, :, :], total_step, dataformats='HW')
        writer.add_image('psf far (1st channel)', psf_far.cpu().detach()[0, 0, :, :], total_step, dataformats='HW')

    if save_images:
        os.makedirs(os.path.join(args.result_path, 'logged_images'), exist_ok=True)
        os.makedirs(os.path.join(args.result_path, 'logged_psf_near'), exist_ok=True)
        os.makedirs(os.path.join(args.result_path, 'logged_psf_far'), exist_ok=True)
        os.makedirs(os.path.join(args.result_path, 'logged_phase_map'), exist_ok=True)

        plt.figure(); plt.imshow(np.clip(psf_near.cpu().detach().numpy()[0, 0, :, :], 0, 1)); plt.colorbar()
        plt.savefig(os.path.join(args.result_path, 'logged_psf_near', f'step_{total_step}.png'), bbox_inches='tight'); plt.clf(); plt.close()
        
        plt.figure(); plt.imshow(np.clip(psf_far.cpu().detach().numpy()[0, 0, :, :], 0, 1)); plt.colorbar()
        plt.savefig(os.path.join(args.result_path, 'logged_psf_far', f'step_{total_step}.png'), bbox_inches='tight'); plt.clf(); plt.close()
        plt.imsave(os.path.join(args.result_path, 'logged_psf_far', f'raw_step_{total_step}.png'), np.clip(psf_far.cpu().detach().numpy()[0, 0, :, :], 0, 1))
        plt.imsave(os.path.join(args.result_path, 'logged_psf_far', f'raw_center_step_{total_step}.png'), np.clip(psf_far.cpu().detach().numpy()[0, 0, psf_far.shape[-2]//2-50:psf_far.shape[-2]//2+50, psf_far.shape[-1]//2-50:psf_far.shape[-1]//2+50], 0, 1))

        plt.figure(); plt.imshow(np.clip(psf_near.cpu().detach().numpy()[0, 0, :, :], 0, 1)); plt.colorbar()
        plt.savefig(os.path.join(args.result_path, 'logged_psf_near', f'step_{total_step}.png'), bbox_inches='tight'); plt.clf(); plt.close()
        plt.figure(); plt.imshow(np.clip(psf_far.cpu().detach().numpy()[0, 0, :, :], 0, 1)); plt.colorbar()
        plt.savefig(os.path.join(args.result_path, 'logged_psf_far', f'step_{total_step}.png'), bbox_inches='tight'); plt.clf(); plt.close()

        if args.use_network:
            plt.figure(); plt.imshow(np.clip(img_conv_before_net.cpu().detach().numpy()[0, :, :, :].transpose(1, 2, 0), 0, 1))
            plt.savefig(os.path.join(args.result_path, 'logged_images', f'step_{total_step}_before_network.png'), bbox_inches='tight'); plt.clf(); plt.close()

        # plt.figure(); plt.imshow(np.clip(img_conv.cpu().detach().numpy()[0, :, :, :].transpose(1, 2, 0), 0, 1))
        # plt.savefig(os.path.join(args.result_path, 'logged_images', f'step_{total_step}.png'), bbox_inches='tight'); plt.clf(); plt.close()
        plt.imsave(os.path.join(args.result_path, 'logged_images', f'raw_step_{total_step}.png'), np.clip(img_conv.cpu().detach().numpy()[0, :, :, :].transpose(1, 2, 0), 0, 1))

        DOE_train = DOE((1,1,param.R, param.R), param.DOE_pitch, param.material, wvl=param.DOE_wvl, device=args.device)
        DOE_train.set_phase_change(DOE_phase)  
        if change_wvl:
            DOE_train.change_wvl(param.DOE_wvl)
        DOE_train.visualize()
        plt.savefig(os.path.join(args.result_path, 'logged_phase_map', f'step_{total_step}.png'), bbox_inches='tight'); plt.clf(); plt.close()

        return l1_loss
    else:
        return (psf_near[0].permute(1, 2, 0).cpu().detach().numpy(),
                psf_far[0].permute(1, 2, 0).cpu().detach().numpy(),
                image_near[0].permute(1, 2, 0).cpu().detach().numpy(),
                image_far[0].permute(1, 2, 0).cpu().detach().numpy(),
                convolved_far[0].permute(1, 2, 0).cpu().detach().numpy(),
                img_conv[0].permute(1, 2, 0).cpu().detach().numpy())
    
#--------- Single sensor synthetic aperture image formation ---------#
def evaluate_image_loss(args, DOE_phase, net, testloader, writer, total_step, single_sensor=False, far_disparity=False, concat=False):
    param = args.param 
    image_far, _ = next(iter(testloader))
    image_far = image_far.to(args.device)
    image_near, mask = compute_dirt(image_far.get_intensity() if isinstance(image_far, pado.light.Light) 
                                        else image_far, (param.depth_near_min+param.depth_near_max)/2, args, predefined=True)
    z_far = (param.depth_far_min+param.depth_far_max)/2
    z_near = (param.depth_near_min+param.depth_near_max)/2
    if args.phase_noise_stddev is not None:
        DOE_phase = DOE_phase + torch.randn((DOE_phase.shape), device=args.device) * args.phase_noise_stddev

    DOE_train = DOE((1,1,param.R, param.R), param.DOE_pitch, param.material, wvl=param.DOE_wvl, device=args.device)
    DOE_train.set_phase_change(DOE_phase.detach()) 

    if single_sensor:
        psf_far = plot_depth_based_psf_single_sensor_synthetic_aperture(DOE_train, args, depths = [(param.depth_far_min+param.depth_far_max)/2], 
                                                                        wvls='RGB', merge_channel=True, 
                                                                        light_source_offset=(param.baseline/2, param.baseline/2), 
                                                                        propagation_offset=(param.baseline/2, param.baseline/2), 
                                                                        normalize=False)
        psf_near = plot_depth_based_psf_single_sensor_synthetic_aperture(DOE_train, args, depths = [(param.depth_near_min+param.depth_near_max)/2], 
                                                                         wvls='RGB', merge_channel=True, 
                                                                        light_source_offset=(param.baseline/2, param.baseline/2), 
                                                                        propagation_offset=(param.baseline/2, param.baseline/2), 
                                                                        normalize=False) 
    else: 
        psf_far, _ = plot_depth_based_psf(DOE_train, args, depths = [z_far], 
                                        wvls='RGB', merge_channel = True, pad=True, eval=True)
        psf_near, _ = plot_depth_based_psf(DOE_train, args, depths = [z_near], 
                                        wvls='RGB', merge_channel = True, pad=True, eval=True)
    normalizer = psf_far[:,1,...].sum()

    psf_near = psf_near / normalizer
    psf_far = psf_far / normalizer

    # convolved_far = fft_convolve2d(image_far.get_intensity() if isinstance(image_far, pado.light.Light) else image_far, 
    #                             psf_far)
    convolved_far = fft_convolve2d_image_reflection(image_far.get_intensity() if isinstance(image_far, pado.light.Light) else image_far, 
                                psf_far)

    # Adjust brightness
    scale_factor_brightness = torch.clamp(image_far.mean() / convolved_far.mean(), max=args.brightness_clamp)
    psf_far = psf_far * scale_factor_brightness
    psf_near = psf_near * scale_factor_brightness
    convolved_far = convolved_far * scale_factor_brightness

    # Compute fixed near obstruction image and mask
    convolved_near = fft_convolve2d(image_near, psf_near)
    # convolved_mask = fft_convolve2d(mask, psf_near)
    convolved_mask = torch.clamp(1.5 * fft_convolve2d(mask, psf_near), 0, 1)

    # Compute loss
    step_loss = 0
    step_psnr = 0
    for step, batch_data in enumerate(testloader):
        image_far_color, _ = batch_data
        image_far_color = image_far_color.to(args.device)

        if param.metasurface_array is not None:
            if concat:
                img_convs = list()
            else:
                img_convs = torch.zeros_like(image_far)
            disparity = round((param.focal_length/param.camera_pitch) * (param.baseline) / ((param.depth_near_min+param.depth_near_max)/2))
            disparities_y, disparities_x = np.meshgrid(np.linspace(-(param.metasurface_array[0]-1)/2*disparity, (param.metasurface_array[0]-1)/2*disparity, param.metasurface_array[0]), 
                                                np.linspace(-(param.metasurface_array[1]-1)/2*disparity, (param.metasurface_array[1]-1)/2*disparity, param.metasurface_array[1]), indexing='ij')
            for disparity_y_, disparity_x_ in zip(disparities_y.ravel(), disparities_x.ravel()):
                disparity_y_ = round(disparity_y_.item())
                disparity_x_ = round(disparity_x_.item())
                convolved_near = shift_image(fft_convolve2d(image_near, psf_near), disparity_y_, disparity_x_)
                convolved_mask = shift_image(fft_convolve2d(mask, psf_near), disparity_y_, disparity_x_)
                convolved_mask = torch.clamp(1.5 * convolved_mask, 0, 1)

                if far_disparity:
                    img_conv = shift_image(convolved_far, round(disparity_y_/(z_far/z_near)), 
                                            round(disparity_x_/(z_far/z_near))) * (1 - convolved_mask) + convolved_near * convolved_mask
                else:
                    img_conv = convolved_far * (1 - convolved_mask) + convolved_near * convolved_mask

                if concat:
                    img_convs.append(img_conv)
                else:
                    img_convs = img_convs + img_conv
            if concat:
                img_conv = torch.cat(img_convs, dim=-3)
            else:
                img_conv = img_convs / len(disparities_y.ravel())
        else:
            img_conv = convolved_far * (1 - convolved_mask) + convolved_near * convolved_mask

        if args.image_noise_stddev is not None:
            img_conv = img_conv + (torch.randn((img_conv.shape), device=args.device) * args.image_noise_stddev).detach()
           
        if net:
            if concat:
                img_conv_before_network = torch.stack([(img_conv[:,0,:,:]+img_conv[:,3,:,:]+img_conv[:,6,:,:]+img_conv[:,9,:,:])/4, 
                                                    (img_conv[:,1,:,:]+img_conv[:,4,:,:]+img_conv[:,7,:,:]+img_conv[:,10,:,:])/4,
                                                    (img_conv[:,2,:,:]+img_conv[:,5,:,:]+img_conv[:,8,:,:]+img_conv[:,11,:,:])/4], dim=-3).clone()
            else:
                img_conv_before_network = img_conv.clone()
            img_conv = net(img_conv)
        
        if step==0:
            os.makedirs(os.path.join(args.result_path, 'logged_images'), exist_ok=True)
            os.makedirs(os.path.join(args.result_path, 'logged_psf_near'), exist_ok=True)
            os.makedirs(os.path.join(args.result_path, 'logged_psf_far'), exist_ok=True)
            os.makedirs(os.path.join(args.result_path, 'logged_phase_map'), exist_ok=True)

            plt.figure(); plt.imshow(np.clip(psf_near.cpu().detach().numpy()[0, 0, :, :], 0, 1)); plt.colorbar()
            plt.savefig(os.path.join(args.result_path, 'logged_psf_near', f'step_{total_step}.png'), bbox_inches='tight'); plt.clf(); plt.close()
            
            plt.figure(); plt.imshow(np.clip(psf_far.cpu().detach().numpy()[0, 0, :, :], 0, 1)); plt.colorbar()
            plt.savefig(os.path.join(args.result_path, 'logged_psf_far', f'step_{total_step}.png'), bbox_inches='tight'); plt.clf(); plt.close()
            plt.imsave(os.path.join(args.result_path, 'logged_psf_far', f'raw_step_{total_step}.png'), np.clip(psf_far.cpu().detach().numpy()[0, 0, :, :], 0, 1))
            plt.imsave(os.path.join(args.result_path, 'logged_psf_far', f'raw_center_step_{total_step}.png'), np.clip(psf_far.cpu().detach().numpy()[0, 0, psf_far.shape[-2]//2-50:psf_far.shape[-2]//2+50, psf_far.shape[-1]//2-50:psf_far.shape[-1]//2+50], 0, 1))

            plt.figure(); plt.imshow(np.clip(psf_near.cpu().detach().numpy()[0, 0, :, :], 0, 1)); plt.colorbar()
            plt.savefig(os.path.join(args.result_path, 'logged_psf_near', f'step_{total_step}.png'), bbox_inches='tight'); plt.clf(); plt.close()
            plt.figure(); plt.imshow(np.clip(psf_far.cpu().detach().numpy()[0, 0, :, :], 0, 1)); plt.colorbar()
            plt.savefig(os.path.join(args.result_path, 'logged_psf_far', f'step_{total_step}.png'), bbox_inches='tight'); plt.clf(); plt.close()

            if args.use_network:
                plt.figure(); plt.imshow(np.clip(img_conv_before_network.cpu().detach().numpy()[0, ...].transpose(1, 2, 0), 0, 1))
                plt.savefig(os.path.join(args.result_path, 'logged_images', f'step_{total_step}_before_network.png'), bbox_inches='tight'); plt.clf(); plt.close()
            plt.imsave(os.path.join(args.result_path, 'logged_images', f'raw_step_{total_step}.png'), np.clip(img_conv.cpu().detach().numpy()[0, :, :, :].transpose(1, 2, 0), 0, 1))

            DOE_train.visualize()
            plt.savefig(os.path.join(args.result_path, 'logged_phase_map', f'step_{total_step}.png'), bbox_inches='tight'); plt.clf(); plt.close()

        step_loss = step_loss + args.l1_criterion(img_conv, image_far).item()
        step_psnr = step_psnr +  skimage.metrics.peak_signal_noise_ratio(
                                image_far[0].permute(1, 2, 0).cpu().detach().numpy(), img_conv[0].permute(1, 2, 0).cpu().detach().numpy())

    l1_loss = step_loss/len(testloader)
    psnr = step_psnr/len(testloader)

    if writer is not None:       
        writer.add_scalar('Loss/train', l1_loss, total_step)  # Log loss for TensorBoard
        writer.add_scalar('PSNR/train', psnr, total_step)
        if args.use_da_loss:
            from models.DA_loss_functions import DA_loss
            da_loss = DA_loss(img_conv, image_far, args).detach().item()
            writer.add_scalar('DA_loss/train', da_loss, total_step)
        writer.add_image('image near', image_near[0].cpu().detach(), total_step)
        writer.add_image('image_far', image_far[0].cpu().detach(), total_step)
        writer.add_image('convolved far only', convolved_far[0].cpu().detach(), total_step)
        writer.add_image('convolved image', img_conv[0].cpu().detach(), total_step)
        writer.add_image('psf near (1st channel)', psf_near.cpu().detach()[0, 0, :, :], total_step, dataformats='HW')
        writer.add_image('psf far (1st channel)', psf_far.cpu().detach()[0, 0, :, :], total_step, dataformats='HW')

    return l1_loss
    
def full_psf(args, DOE_phase, theta_y, theta_x, normalizer=None, RGB=True, change_wvl=False):
    # Mostly used for inference

    param = args.param

    z_near = (param.depth_near_max + param.depth_near_min)/2
    z_far = (param.depth_far_max + param.depth_far_min)/2

    theta = (theta_y, theta_x)
    full_wavefield_dim_R = param.full_psf_scale[0] * param.R
    full_wavefield_dim_C = param.full_psf_scale[1] * param.C
    _, _, full_psf_dim_R, full_psf_dim_C = F.interpolate(torch.zeros((1, 1, full_wavefield_dim_R, full_wavefield_dim_C)), scale_factor=param.DOE_pitch/param.camera_pitch).detach().shape
    
    wvls = param.wvls if RGB else [param.DOE_wvl]
    full_psf_near_shifted = torch.zeros((1, len(wvls), full_psf_dim_R, full_psf_dim_C), device=args.device)
    full_psf_far_shifted = torch.zeros((1, len(wvls), full_psf_dim_R, full_psf_dim_C), device=args.device)
    full_psf_far = torch.zeros((1, len(wvls), full_psf_dim_R, full_psf_dim_C), device=args.device)
    full_psf_near = torch.zeros((1, len(wvls), full_psf_dim_R, full_psf_dim_C), device=args.device)

    if args.phase_noise_stddev is not None:
        DOE_phase = DOE_phase + torch.randn((DOE_phase.shape), device=args.device) * args.phase_noise_stddev

    for channel_idx, wvl in enumerate(wvls):
        DOE_train = DOE((1,1,param.R, param.R), param.DOE_pitch, param.material, wvl=param.DOE_wvl, device=args.device)
        DOE_train.set_phase_change(DOE_phase)
        # debugging
        if change_wvl:
            DOE_train.change_wvl(param.DOE_wvl) # will be debugged
        # debugging
        # Far
        full_psf_far_ = compute_psf_arbitrary_prop(wvl, 
                                        torch.tensor(z_far) if not isinstance(z_far, torch.Tensor) else z_far.clone().detach(), 
                                        DOE_train, 
                                        args, 
                                        propagator=args.propagator,
                                        use_lens=args.use_lens,
                                        offset=(0, 0),
                                        variable_offset_indices=[],
                                        theta=theta,
                                        pad=True,
                                        full_psf=True,
                                        normalize=False)
        # Near
        full_psf_near_ = compute_psf_arbitrary_prop(wvl, 
                                        torch.tensor(z_near) if not isinstance(z_near, torch.Tensor) else z_near.clone().detach(), 
                                        DOE_train, 
                                        args, 
                                        propagator=args.propagator,
                                        use_lens=args.use_lens,
                                        offset=(0, 0),
                                        variable_offset_indices=[],
                                        theta=theta,
                                        pad=True,
                                        full_psf=True,
                                        normalize=False)
    
        # Normalize
        if normalizer is None:
            normalizer = 1
        full_psf_near[:, channel_idx, :, :] = full_psf_near_ / normalizer
        full_psf_far[:, channel_idx, :, :] = full_psf_far_ / normalizer

    # Shift PSF based on the position of the PSF peak.
    _, _, h, w = full_psf_far.shape
    c = (full_psf_far == full_psf_far.max()).nonzero(as_tuple=False)[0]
    shift_y, shift_x = h//2-c[2], w//2-c[3]
    if shift_x >= 0 and shift_y >= 0:
        full_psf_near_shifted[:,:,shift_y:, shift_x:] = full_psf_near[:, :, :None if shift_y==0 else -shift_y, :None if shift_x==0 else -shift_x]
        full_psf_far_shifted[:,:,shift_y:, shift_x:] = full_psf_far[:, :, :None if shift_y==0 else -shift_y, :None if shift_x==0 else -shift_x]
    elif shift_x < 0 and shift_y >= 0:
        full_psf_near_shifted[:,:,shift_y:, :shift_x] = full_psf_near[:, :, :None if shift_y==0 else -shift_y, None if shift_x==0 else -shift_x:]
        full_psf_far_shifted[:,:,shift_y:, :shift_x] = full_psf_far[:, :, :None if shift_y==0 else -shift_y, None if shift_x==0 else -shift_x:]
    elif shift_x >=0 and shift_y < 0:
        full_psf_near_shifted[:,:,:shift_y, shift_x:] = full_psf_near[:, :, None if shift_y==0 else -shift_y:, :None if shift_x==0 else -shift_x]
        full_psf_far_shifted[:,:,:shift_y, shift_x:] = full_psf_far[:, :, None if shift_y==0 else -shift_y:, :None if shift_x==0 else -shift_x]
    else: 
        full_psf_near_shifted[:,:,:shift_y, :shift_x] = full_psf_near[:, :, None if shift_y==0 else -shift_y:, None if shift_x==0 else -shift_x:]
        full_psf_far_shifted[:,:,:shift_y, :shift_x] = full_psf_far[:, :, None if shift_y==0 else -shift_y:, None if shift_x==0 else -shift_x:]

    return full_psf_near_shifted, full_psf_far_shifted

def evaluate_DOE(DOE_phase, total_step, args, writer=None, net=None, save_images=True, RGB=False, obstruction_reinforcement=False, phase_noise=False, image_noise=False, vis_psf=False, simple_psf=False, change_wvl=False, adjust=False):
    # Evaluate DOE using predetermined mask and obstacle image
    param = args.param
    
    if phase_noise and args.phase_noise_stddev is not None:
        DOE_phase = DOE_phase + torch.randn((DOE_phase.shape), device=args.device) * args.phase_noise_stddev

    image_near = torch.tensor(np.load(os.path.join(args.eval_path, 'img_near_'+str(param.img_res)+'_0022.npy')), device=args.device).detach()
    image_far = torch.tensor(np.load(os.path.join(args.eval_path, 'img_far_'+str(param.img_res)+'_0022.npy')), device=args.device).detach()
    mask = torch.tensor(np.load(os.path.join(args.eval_path, 'mask_'+str(param.img_res)+'_0022.npy')), device=args.device).detach()

    if simple_psf:
        DOE_train = DOE((1,1,param.R, param.R), param.DOE_pitch, param.material, wvl=param.DOE_wvl, device=args.device)
        DOE_train.set_phase_change(DOE_phase.detach())  
        if change_wvl:
            DOE_train.change_wvl(param.DOE_wvl)
        psf_far, _ = plot_depth_based_psf(DOE_train, args, depths = [(param.depth_far_min+param.depth_far_max)/2], 
                                        wvls='RGB' if RGB is True else param.DOE_wvl, 
                                        merge_channel = True, pad=True, eval=True)
        psf_near, _ = plot_depth_based_psf(DOE_train, args, depths = [(param.depth_near_min+param.depth_near_max)/2], 
                                        wvls='RGB' if RGB is True else param.DOE_wvl, 
                                        merge_channel = True, pad=True, eval=True)
    else:
        psf_near, psf_far = full_psf(args, DOE_phase.detach(), 0, 0, normalizer=None, RGB=RGB, change_wvl=change_wvl)

    if args.normalizing_method == 'new' and (psf_far.shape[-2] > param.camera_resolution[-2] and psf_far.shape[-1] > param.camera_resolution[-1]): # approximate
        normalizer = psf_far[:, 1 if RGB else 0, psf_far.shape[-2]//2-param.camera_resolution[-2]//2:psf_far.shape[-2]//2+param.camera_resolution[-2]//2,
                                psf_far.shape[-1]//2-param.camera_resolution[-1]//2:psf_far.shape[-1]//2+param.camera_resolution[-1]//2].sum()
    else:
        normalizer = psf_far[:,1,...].sum() if RGB else psf_far.sum()

    psf_near = psf_near / normalizer
    psf_far = psf_far / normalizer

    convolved_far = fft_convolve2d(image_far.get_intensity() if isinstance(image_far, pado.light.Light) else image_far, 
                                psf_far)

    # Adjust brightness
    scale_factor_brightness = torch.clamp(image_far.mean() / convolved_far.mean(), max=args.brightness_clamp)
    print('brightness scale factor:', scale_factor_brightness)
    psf_far = psf_far * scale_factor_brightness
    psf_near = psf_near * scale_factor_brightness
    convolved_far = convolved_far * scale_factor_brightness

    log_psf_near = torch.log(psf_near + 1e-9)
    log_psf_near -= torch.min(log_psf_near)
    log_psf_near /= torch.max(log_psf_near)

    log_psf_far = torch.log(psf_far + 1e-9)
    log_psf_far -= torch.min(log_psf_far)
    log_psf_far /= torch.max(log_psf_far)

    if vis_psf:
        if psf_near.shape[1]==1: 
            colors = ['grayscale']
        else:
            colors = ['Red', 'Green', 'Blue']
        for c, color in enumerate(colors):
            plt.figure(figsize=(10, 10))
            plt.title(f'PSF near {color}')
            plt.imshow(psf_near[0, c, ...].cpu().numpy(), vmax=0.0001) #  vmax=0.00001
            plt.colorbar()
            plt.figure(figsize=(10, 10))
            plt.title(f'PSF far {color}')
            plt.imshow(psf_far[0, c, ...].cpu().numpy(), vmax=0.001)
            plt.colorbar()

            plt.figure(figsize=(10, 10))
            plt.title(f'PSF near log {color}')
            plt.imshow(log_psf_near[0, c, ...].cpu().numpy())
            plt.colorbar()
            plt.figure(figsize=(10, 10))
            plt.title(f'PSF far log{color}')
            plt.imshow(log_psf_far[0, c, ...].cpu().numpy())
            plt.colorbar()

            plt.figure(figsize=(5, 5))
            plt.imshow(psf_near[0, c, ...][psf_near.shape[2]//2-30:psf_near.shape[2]//2+30, psf_near.shape[3]//2-30:psf_near.shape[3]//2+30].cpu().numpy())
            plt.colorbar()
            plt.title(f'PSF_near_center {color}')
            plt.figure(figsize=(5, 5))
            plt.imshow(psf_far[0, c, ...][psf_far.shape[2]//2-30:psf_far.shape[2]//2+30, psf_far.shape[3]//2-30:psf_far.shape[3]//2+30].cpu().numpy())
            plt.colorbar()
            plt.title(f'PSF_far_center {color}')

            airy_disk_diameter = round(2.44*param.wvls[c]*(param.focal_length/(param.DOE_pitch*param.R))/param.camera_pitch)
            ylim=psf_far.max().cpu().numpy()*airy_disk_diameter*3

            plt.figure(figsize=(13, 5))
            graph_1d = psf_far[0, c, ...].cpu().numpy()
            plt.plot(graph_1d.sum(axis=0))
            plt.ylim([0, ylim])
            plt.title(f"""PSF far {color} 1D visualization, 
peak sum={graph_1d.sum(axis=0)[psf_far.shape[-1]//2-airy_disk_diameter:psf_far.shape[-1]//2+airy_disk_diameter].sum():.5f},
peak max={graph_1d.sum(axis=0).max():.5f}""")
            plt.show()

            plt.figure(figsize=(13, 5))
            graph_1d = psf_near[0, c, ...].cpu().numpy()
            plt.plot(graph_1d.sum(axis=0))
            plt.ylim([0, ylim])
            plt.title(f"""PSF near {color} 1D visualization, 
peak sum={graph_1d.sum(axis=0)[psf_far.shape[-1]//2-airy_disk_diameter:psf_far.shape[-1]//2+airy_disk_diameter].sum():.5f},
peak max={graph_1d.sum(axis=0).max():.5f}""")
            plt.show()

            ylim=psf_far.max().cpu().numpy()*airy_disk_diameter*2

            plt.figure(figsize=(13, 5))
            graph_1d = psf_far[0, c, psf_far.shape[-2]//2-airy_disk_diameter:psf_far.shape[-2]//2+airy_disk_diameter, :].cpu().numpy()
            plt.plot(graph_1d.sum(axis=0))
            plt.ylim([0, ylim])
            plt.title(f"""PSF far {color} center 1D visualization, 
peak sum={graph_1d.sum(axis=0)[psf_far.shape[-1]//2-airy_disk_diameter:psf_far.shape[-1]//2+airy_disk_diameter].sum():.5f},
peak max={graph_1d.sum(axis=0).max():.5f}""")
            plt.show()

            plt.figure(figsize=(13, 5))
            graph_1d = psf_near[0, c, psf_far.shape[-2]//2-airy_disk_diameter:psf_far.shape[-2]//2+airy_disk_diameter, :].cpu().numpy()
            plt.plot(graph_1d.sum(axis=0))
            plt.ylim([0, ylim])
            plt.title(f"""PSF near {color} center 1D visualization, 
peak sum={graph_1d.sum(axis=0)[psf_far.shape[-1]//2-airy_disk_diameter:psf_far.shape[-1]//2+airy_disk_diameter].sum():.5f},
peak max={graph_1d.sum(axis=0).max():.5f}""")
            plt.show()

            print(f'channel {c}, psf near center sum: {psf_near[0, c, ...][psf_near.shape[2]//2-5:psf_near.shape[2]//2+5, psf_near.shape[3]//2-5:psf_near.shape[3]//2+5].sum()}', 
                  f'channel {c}, psf far center sum: {psf_far[0, c, ...][psf_far.shape[2]//2-5:psf_far.shape[2]//2+5, psf_far.shape[3]//2-5:psf_far.shape[3]//2+5].sum()}')
  
    if param.metasurface_array is not None:
        img_convs = torch.zeros_like(image_near)
        disparity = round((param.focal_length/param.camera_pitch) * (param.baseline) / ((param.depth_near_min+param.depth_near_max)/2))
        disparities_y, disparities_x = np.meshgrid(np.linspace(-(param.metasurface_array[0]-1)/2*disparity, (param.metasurface_array[0]-1)/2*disparity, param.metasurface_array[0]), 
                                            np.linspace(-(param.metasurface_array[1]-1)/2*disparity, (param.metasurface_array[1]-1)/2*disparity, param.metasurface_array[1]), indexing='ij')
        for disparity_y_, disparity_x_ in zip(disparities_y.ravel(), disparities_x.ravel()):
            disparity_y_ = round(disparity_y_.item())
            disparity_x_ = round(disparity_x_.item())
            convolved_near = shift_image(fft_convolve2d(image_near, psf_near), disparity_y_, disparity_x_)
            convolved_mask = shift_image(fft_convolve2d(mask, psf_near), disparity_y_, disparity_x_)
            convolved_mask = torch.clamp(1.5 * convolved_mask, 0, 1)
            img_conv = convolved_far * (1 - convolved_mask) + convolved_near * convolved_mask
            img_convs = img_convs + img_conv
        img_conv = img_convs / len(disparities_y.ravel())
    else:
        convolved_near = fft_convolve2d(image_near, psf_near)
        convolved_mask = fft_convolve2d(mask, psf_near)
        convolved_mask = torch.clamp(1.5 * convolved_mask, 0, 1)
        img_conv = convolved_far * (1 - convolved_mask) + convolved_near * convolved_mask

    if args.use_network:
        img_conv_before_net = img_conv.clone()
        img_conv = net(img_conv.to(torch.float32))
    
    # Adjust contrast
    scale_factor_contrast = torch.clamp(image_far.std()/img_conv.std(), max=args.contrast_clamp).item()
    img_conv = img_conv.mean() + scale_factor_contrast * (img_conv - img_conv.mean())

    if adjust:
        img_conv = adjust_brightness_and_contrast(img_conv, image_far)

    if image_noise and args.image_noise_stddev is not None:
        noise = torch.randn((img_conv.shape), device=args.device) * args.image_noise_stddev
        img_conv = torch.clamp(img_conv + noise.type_as(img_conv), 0, 1)
        convolved_far = torch.clamp(convolved_far + noise.type_as(img_conv), 0, 1)
    
    psnr = skimage.metrics.peak_signal_noise_ratio(
        image_far[0].permute(1, 2, 0).cpu().detach().numpy(), img_conv[0].permute(1, 2, 0).cpu().detach().numpy())
    l1_loss = args.l1_criterion(img_conv, image_far).detach().item()

    print('l1_loss: ', l1_loss, 'psnr: ', psnr, 'brightness multiplier: ', scale_factor_brightness)

    if writer is not None:       
        writer.add_scalar('Loss/train', l1_loss, total_step)  # Log loss for TensorBoard
        writer.add_scalar('PSNR/train', psnr, total_step)
        if args.use_da_loss:
            from models.DA_loss_functions import DA_loss
            da_loss = DA_loss(img_conv, image_far, args).detach().item()
            writer.add_scalar('DA_loss/train', da_loss, total_step)
        writer.add_image('image near', image_near[0].cpu().detach(), total_step)
        writer.add_image('image_far', image_far[0].cpu().detach(), total_step)
        writer.add_image('convolved far only', convolved_far[0].cpu().detach(), total_step)
        writer.add_image('convolved image', img_conv[0].cpu().detach(), total_step)
        writer.add_image('psf near (1st channel)', psf_near.cpu().detach()[0, 0, :, :], total_step, dataformats='HW')
        writer.add_image('psf far (1st channel)', psf_far.cpu().detach()[0, 0, :, :], total_step, dataformats='HW')

    if save_images:
        os.makedirs(os.path.join(args.result_path, 'logged_images'), exist_ok=True)
        os.makedirs(os.path.join(args.result_path, 'logged_psf_near'), exist_ok=True)
        os.makedirs(os.path.join(args.result_path, 'logged_psf_far'), exist_ok=True)
        os.makedirs(os.path.join(args.result_path, 'logged_phase_map'), exist_ok=True)

        plt.figure(); plt.imshow(np.clip(psf_near.cpu().detach().numpy()[0, 0, :, :], 0, 1)); plt.colorbar()
        plt.savefig(os.path.join(args.result_path, 'logged_psf_near', f'step_{total_step}.png'), bbox_inches='tight'); plt.clf(); plt.close()
        
        plt.figure(); plt.imshow(np.clip(psf_far.cpu().detach().numpy()[0, 0, :, :], 0, 1)); plt.colorbar()
        plt.savefig(os.path.join(args.result_path, 'logged_psf_far', f'step_{total_step}.png'), bbox_inches='tight'); plt.clf(); plt.close()
        plt.imsave(os.path.join(args.result_path, 'logged_psf_far', f'raw_step_{total_step}.png'), np.clip(psf_far.cpu().detach().numpy()[0, 0, :, :], 0, 1))
        plt.imsave(os.path.join(args.result_path, 'logged_psf_far', f'raw_center_step_{total_step}.png'), np.clip(psf_far.cpu().detach().numpy()[0, 0, psf_far.shape[-2]//2-50:psf_far.shape[-2]//2+50, psf_far.shape[-1]//2-50:psf_far.shape[-1]//2+50], 0, 1))

        plt.figure(); plt.imshow(np.clip(psf_near.cpu().detach().numpy()[0, 0, :, :], 0, 1)); plt.colorbar()
        plt.savefig(os.path.join(args.result_path, 'logged_psf_near', f'step_{total_step}.png'), bbox_inches='tight'); plt.clf(); plt.close()
        plt.figure(); plt.imshow(np.clip(psf_far.cpu().detach().numpy()[0, 0, :, :], 0, 1)); plt.colorbar()
        plt.savefig(os.path.join(args.result_path, 'logged_psf_far', f'step_{total_step}.png'), bbox_inches='tight'); plt.clf(); plt.close()

        if args.use_network:
            plt.figure(); plt.imshow(np.clip(img_conv_before_net.cpu().detach().numpy()[0, :, :, :].transpose(1, 2, 0), 0, 1))
            plt.savefig(os.path.join(args.result_path, 'logged_images', f'step_{total_step}_before_network.png'), bbox_inches='tight'); plt.clf(); plt.close()

        # plt.figure(); plt.imshow(np.clip(img_conv.cpu().detach().numpy()[0, :, :, :].transpose(1, 2, 0), 0, 1))
        # plt.savefig(os.path.join(args.result_path, 'logged_images', f'step_{total_step}.png'), bbox_inches='tight'); plt.clf(); plt.close()
        plt.imsave(os.path.join(args.result_path, 'logged_images', f'raw_step_{total_step}.png'), np.clip(img_conv.cpu().detach().numpy()[0, :, :, :].transpose(1, 2, 0), 0, 1))

        DOE_train = DOE((1,1,param.R, param.R), param.DOE_pitch, param.material, wvl=param.DOE_wvl, device=args.device)
        DOE_train.set_phase_change(DOE_phase)  
        if change_wvl:
            DOE_train.change_wvl(param.DOE_wvl)
        DOE_train.visualize()
        plt.savefig(os.path.join(args.result_path, 'logged_phase_map', f'step_{total_step}.png'), bbox_inches='tight'); plt.clf(); plt.close()

        return l1_loss
    else:
        return (psf_near[0].permute(1, 2, 0).cpu().detach().numpy(),
                psf_far[0].permute(1, 2, 0).cpu().detach().numpy(),
                image_near[0].permute(1, 2, 0).cpu().detach().numpy(),
                image_far[0].permute(1, 2, 0).cpu().detach().numpy(),
                convolved_far[0].permute(1, 2, 0).cpu().detach().numpy(),
                img_conv[0].permute(1, 2, 0).cpu().detach().numpy())
    
def evaluate_DOE_svconv(DOE_phase, args, net=None, RGB=False, obstruction_reinforcement=False, phase_noise=False, image_noise=False, change_wvl=False, adjust=False):
    # Evaluate DOE using predetermined mask and obstacle image
    param = args.param
    
    if phase_noise and args.phase_noise_stddev is not None:
        DOE_phase = DOE_phase + torch.randn((DOE_phase.shape), device=args.device) * args.phase_noise_stddev

    image_near = torch.tensor(np.load(os.path.join(args.eval_path, 'img_near_'+str(param.img_res)+'_0022.npy')), device=args.device).detach()
    image_far = torch.tensor(np.load(os.path.join(args.eval_path, 'img_far_'+str(param.img_res)+'_0022.npy')), device=args.device).detach()
    mask = torch.tensor(np.load(os.path.join(args.eval_path, 'mask_'+str(param.img_res)+'_0022.npy')), device=args.device).detach()


    center_psf_near, center_psf_far = full_psf(args, DOE_phase, 0, 0, normalizer=None, RGB=RGB, change_wvl=change_wvl)

    if args.normalizing_method == 'new' and (center_psf_far.shape[-2] > param.camera_resolution[-2] and center_psf_far.shape[-1] > param.camera_resolution[-1]): # approximate
        normalizer = center_psf_far[:, 1 if RGB else 0, center_psf_far.shape[-2]//2-param.camera_resolution[-2]//2:center_psf_far.shape[-2]//2+param.camera_resolution[-2]//2,
                            center_psf_far.shape[-1]//2-param.camera_resolution[-1]//2:center_psf_far.shape[-1]//2+param.camera_resolution[-1]//2].sum()
    else:
        normalizer = center_psf_far[:,1,...].sum() if RGB else center_psf_far.sum()

    convolved_far = fft_convolve2d(image_far.get_intensity().detach() if isinstance(image_far, pado.light.Light) else image_far, 
                            center_psf_far.detach())
    scale_factor_brightness = torch.clamp(image_far.mean() / convolved_far.mean(), max=args.brightness_clamp)
    print('Eval SvConv, brightness scale factor:', scale_factor_brightness)
    normalizer = normalizer / scale_factor_brightness

    spatially_varying_image = torch.zeros_like(image_far)
    spatially_varying_convolved_far = torch.zeros_like(image_far)

    sv_res = 4
    patch_size_y = param.camera_resolution[0] // sv_res
    patch_size_x = param.camera_resolution[1] // sv_res
    sensor_bw_y = param.camera_resolution[0]*param.camera_pitch
    sensor_bw_x = param.camera_resolution[1]*param.camera_pitch
    max_angle_y = np.rad2deg(np.arctan(sensor_bw_y * ( (sv_res-1) / sv_res) / 2 / param.focal_length))
    max_angle_x = np.rad2deg(np.arctan(sensor_bw_x * ( (sv_res-1) / sv_res) / 2 / param.focal_length))
    theta_list_y, theta_list_x = np.meshgrid(np.linspace(-max_angle_y, max_angle_y, 4), np.linspace(-max_angle_x, max_angle_x, 4), indexing='ij')
    idx_list_y, idx_list_x = np.meshgrid(np.arange(sv_res-1, -1, -1), np.arange(sv_res-1, -1, -1), indexing='ij') # just because tan(10degree): tan(20degree) ~= 1:2

    for (theta_y, theta_x), (idx_y, idx_x) in zip(zip(theta_list_y.ravel(), theta_list_x.ravel()), zip(idx_list_y.ravel(), idx_list_x.ravel())):
        psf_near_shifted, psf_far_shifted = full_psf(args, DOE_phase, theta_y.item(), theta_x.item(), normalizer=normalizer, RGB=RGB, change_wvl=change_wvl)

        convolved_far = fft_convolve2d(image_far, psf_far_shifted.to(args.device))
        convolved_near = fft_convolve2d(image_near, psf_near_shifted.to(args.device))
        convolved_mask = fft_convolve2d(mask, psf_near_shifted.to(args.device))

        if obstruction_reinforcement is True:
            convolved_mask = torch.clamp(1.5*convolved_mask, 0, 1)

        img_conv = convolved_far * (1 - convolved_mask) + convolved_near * convolved_mask

        spatially_varying_image[:, :, idx_y*patch_size_y:(idx_y+1)*patch_size_y, 
                                idx_x*patch_size_x:(idx_x+1)*patch_size_x] = img_conv[:, :,  idx_y*patch_size_y:(idx_y+1)*patch_size_y, 
                                                 idx_x*patch_size_x:(idx_x+1)*patch_size_x]
        spatially_varying_convolved_far[:, :, idx_y*patch_size_y:(idx_y+1)*patch_size_y, 
                                idx_x*patch_size_x:(idx_x+1)*patch_size_x] = convolved_far[:, :,  idx_y*patch_size_y:(idx_y+1)*patch_size_y, 
                                                 idx_x*patch_size_x:(idx_x+1)*patch_size_x]
        
        
    if args.use_network:
        img_conv_before_net = spatially_varying_image.clone()
        spatially_varying_image = net(img_conv_before_net.to(torch.float32))

    if image_noise and args.image_noise_stddev is not None:
        noise = torch.randn((spatially_varying_image.shape), device=args.device) * args.image_noise_stddev
        spatially_varying_image = torch.clamp(spatially_varying_image + noise.type_as(spatially_varying_image), 0, 1)
        convolved_far = torch.clamp(convolved_far + noise.type_as(spatially_varying_image), 0, 1)
    
    return (image_near[0].permute(1, 2, 0).cpu().detach().numpy(),
            image_far[0].permute(1, 2, 0).cpu().detach().numpy(),
            spatially_varying_convolved_far[0].permute(1, 2, 0).cpu().detach().numpy(),
            spatially_varying_image[0].permute(1, 2, 0).cpu().detach().numpy())

# Utility and Miscellaneous Functions

def randuni(low, high, size):
    '''uniformly sample from [low, high)'''
    return (torch.rand(size)*(high - low) + low)

def compute_pad_size(current_size, target_size):
    if current_size == target_size:
        return (0, 0)
    assert current_size < target_size
    gap = target_size - current_size
    left = int(gap/2)
    right = gap - left
    return int(left), int(right)

def edge_mask(R,cutoff, device):
    [x, y] = np.mgrid[-int(R):int(R),-int(R):int(R)]
    dist = np.sqrt(x**2 +y**2).astype(np.int32)
    mask = torch.tensor(1.0*(dist < cutoff)).to(torch.float32)[None, None, ...]
    return mask.to(device)

def ring_mask(R, C, ring_radi, ring_width, device):
    x, y = torch.meshgrid(torch.arange(-R//2, R//2).to(torch.float32).to(device), 
                          torch.arange(-C//2, C//2).to(torch.float32).to(device), indexing='xy')
    radi = x * x + y * y
    
    return ((radi < ring_radi**2) & (radi > (ring_radi-ring_width)**2)).unsqueeze(0).unsqueeze(0)

def adjust_brightness_and_contrast(image, target_image):
    """
    Adjusts the brightness and contrast of each channel of the input RGB image to match the target RGB image.
    
    Parameters:
    - image: Input RGB image as a NumPy array or PyTorch tensor.
    - target_image: Target RGB image as a NumPy array or PyTorch tensor.
    
    Returns:
    - Adjusted RGB image as the same type (NumPy array or PyTorch tensor) as the input.
    """

    def _adjust_brightness_and_contrast_one_channel(image, target_mean, target_std):
        mean = image.mean()
        std = image.std()

        # Adjust brightness and contrast
        adjusted_image = (image - mean) * (target_std / std) + target_mean
        adjusted_image = adjusted_image.clamp(0, 1) if isinstance(image, torch.Tensor) else np.clip(adjusted_image, 0, 1)

        return adjusted_image

    # Determine if inputs are NumPy arrays or PyTorch tensors
    is_tensor = isinstance(image, torch.Tensor)
    if is_tensor:
        adjusted_image = torch.zeros_like(image)
    else:
        adjusted_image = np.zeros_like(image, dtype=np.float64)

    for channel in range(3):  # Iterate over R, G, B channels
        if is_tensor:
            target_mean = target_image[:, channel, :, :].mean()
            target_std = target_image[:, channel, :, :].std()
            adjusted_image[:, channel, :, :] = _adjust_brightness_and_contrast_one_channel(image[:, channel, :, :], target_mean, target_std)
        else:
            target_mean = np.mean(target_image[:, :, channel])
            target_std = np.std(target_image[:, :, channel])
            adjusted_image[:, :, channel] = _adjust_brightness_and_contrast_one_channel(image[:, :, channel], target_mean, target_std)

    return adjusted_image

def add_poisson_noise(img: torch.Tensor, peak: float = 1.0) -> torch.Tensor:
    """
    Add Poisson noise to an image.

    Args:
        img (torch.Tensor): Input image tensor of shape (..., H, W) or (..., C, H, W),
                            with non-negative values.
        peak (float):       Scaling factor that represents the maximum expected photon count.
                            If your image values are in [0, 1], setting peak > 1 simulates
                            brighter images (higher SNR). Default: 1.0.

    Returns:
        torch.Tensor: Noisy image tensor, same shape and dtype as input.
    """
    # Ensure non-negative
    img = torch.clamp(img, min=0)

    # Scale image to photon counts
    vals = img * peak

    # Sample Poisson noise
    noisy_vals = torch.poisson(vals)

    # Scale back to original range
    noisy_img = noisy_vals.to(img.dtype) / peak

    return noisy_img

def visualize_tensor(tensor):
    '''Visualize RGB or single channel tensor.'''
    tensor = tensor[0].detach().numpy()  # Adjust indexing if needed
    if tensor.shape[0] == 3:  # RGB Image
        tensor = np.transpose(tensor, (1, 2, 0))
        plt.figure(figsize=(10, 10))
        plt.imshow(np.clip(tensor, 0, 1))
    else:  # Grayscale Image
        tensor = tensor[0]
        plt.figure(figsize=(10, 10))
        plt.imshow(tensor, cmap='gray')
    plt.title('Visualized Output')
    plt.show()

def visualize_RGB_PSF(psf, title="", emphasize_center=False):
    """Emphasize center option is usedful for Far field PSFs.
    Assume psf.shape is (1, 1, r, c).
    """
    R = psf.shape[-2]
    DOE_psfs_img = torch.cat([psf[0, 0,:,:], psf[0, 1,:,:], psf[0, 2,:,:]], dim=-1).cpu().detach() 
    max_val = DOE_psfs_img.max()
    DOE_psfs_img[:, R:R+1]=max_val
    DOE_psfs_img[:, 2*R:2*R+1]=max_val

    plt.figure(figsize=(24, 8))
    plt.title(title)
    plt.imshow(DOE_psfs_img)
    plt.colorbar()
    plt.show()

    if emphasize_center is True:
        R = psf.shape[-2]
        DOE_psfs_center_img = torch.cat([psf[0, 0, R//2-30:R//2+30, R//2-30:R//2+30], 
                                         psf[0, 1, R//2-30:R//2+30, R//2-30:R//2+30], 
                                         psf[0, 2, R//2-30:R//2+30, R//2-30:R//2+30]], dim=-1).cpu().detach() 
        max_val = DOE_psfs_center_img.max()
        R = 60
        DOE_psfs_center_img[:, R:R+1]=max_val
        DOE_psfs_center_img[:, 2*R:2*R+1]=max_val

        plt.figure(figsize=(24, 8))
        plt.title(title+" center")
        plt.imshow(DOE_psfs_center_img)
        plt.colorbar()
        plt.show()

def visualize_DOE_eval_result(image_near: np.array, image_far: np.array, 
                              convolved_far_DOE:np.array, img_conv_DOE, img_conv_thin_lens,
                              adjust_base='far', save_dir=None):
    """Args:
    adjust_base: if 'far', adjust brightness and contrast of 
                conovlved images based on image_far, else, adjust based on image_near
    save_dir: if noe None, that is, if specified, save images to the specified directory.
    """

    print("image near PSNR: ", skimage.metrics.peak_signal_noise_ratio(
            image_far, image_near))
    print("image far convolved only PSNR: ", skimage.metrics.peak_signal_noise_ratio(
            image_far, convolved_far_DOE))

    plt.figure(figsize=(40, 10))
    fig, axes = plt.subplots(1, 5, figsize=(32, 8))
    axes[0].imshow(image_near)
    axes[0].set_title('Image near', fontsize=20)
    axes[1].imshow(image_far)
    axes[1].set_title('Image far', fontsize=20)
    axes[2].imshow(convolved_far_DOE)
    axes[2].set_title(f'''DOE convolved far only\nPSNR: {skimage.metrics.peak_signal_noise_ratio(
            image_far, convolved_far_DOE)}''', fontsize=20)
    axes[3].imshow(img_conv_DOE)
    axes[3].set_title(f'''DOE image\nPSNR: {skimage.metrics.peak_signal_noise_ratio(
            image_far, img_conv_DOE)}''', fontsize=20)
    axes[4].imshow(img_conv_thin_lens)
    axes[4].set_title(f'''Thin lens image\n PSNR: {skimage.metrics.peak_signal_noise_ratio(
            image_far, img_conv_thin_lens)}''', fontsize=20)
    plt.suptitle(f'dim: {(image_far.shape[-3], image_far.shape[-2])}', fontsize=20)
    plt.tight_layout(rect=[0, 0.2, 1, 0.95])
    plt.show()

    convolved_far_DOE_adjusted = adjust_brightness_and_contrast(convolved_far_DOE, image_far if adjust_base=='far' else image_near)
    img_conv_DOE_adjusted = adjust_brightness_and_contrast(img_conv_DOE, image_far if adjust_base=='far' else image_near)
    plt.figure(figsize=(20, 10))
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    axes[0].imshow(image_far)
    axes[0].set_title('Image far', fontsize=15)
    axes[1].imshow(convolved_far_DOE_adjusted)
    axes[1].set_title(f'''Adj, DOE convolved far only\nPSNR: {skimage.metrics.peak_signal_noise_ratio(
            image_far, convolved_far_DOE_adjusted)}''', fontsize=15)
    axes[2].imshow(img_conv_DOE_adjusted)
    axes[2].set_title(f'''Adj, DOE image\nPSNR: {skimage.metrics.peak_signal_noise_ratio(
            image_far, img_conv_DOE_adjusted)}''', fontsize=15)
    plt.suptitle(f'Brightness, contrast adjusted\ndim: {(image_far.shape[-3], image_far.shape[-2])}', fontsize=18)
    plt.tight_layout(rect=[0, 0.2, 1, 0.95])
    plt.show()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'brightness_adjusted'), exist_ok=True)

        plt.imsave(os.path.join(save_dir, 'obstructed_image(image_near).png'), np.clip(image_near, 0, 1))
        plt.imsave(os.path.join(save_dir, 'clean_image(image_far).png'), np.clip(image_far, 0, 1))
        plt.imsave(os.path.join(save_dir, 'DOE_convolved_clean_only(DOE_convolved_far_only).png'), np.clip(convolved_far_DOE, 0, 1))
        plt.imsave(os.path.join(save_dir, 'DOE_image.png'), np.clip(img_conv_DOE, 0, 1))
        plt.imsave(os.path.join(save_dir, 'Thin_lens_image.png'), np.clip(img_conv_thin_lens, 0, 1))

        plt.imsave(os.path.join(save_dir, 'brightness_adjusted', 'DOE_convolved_clean_only(DOE_convolved_far_only).png'), np.clip(convolved_far_DOE_adjusted, 0, 1))
        plt.imsave(os.path.join(save_dir, 'brightness_adjusted', 'DOE_image.png'), np.clip(img_conv_DOE_adjusted, 0, 1))

def shift_image(image: torch.Tensor, shift_y: int, shift_x: int) -> torch.Tensor:
    _, C, H, W = image.shape

    # Roll the image
    shifted = torch.roll(image, shifts=(shift_y, shift_x), dims=(-2, -1))
    return shifted

def shift_image_with_depth(image: torch.Tensor, baseline: tuple, param, depth=None, depth_map=None) -> torch.Tensor:

    assert (depth is not None) or (depth_map is not None)

    if depth is not None:
        # Roll the image
        disparity_map_y = round(((param.focal_length/param.camera_pitch) * (baseline[0]) / depth).item())
        disparity_map_x = round(((param.focal_length/param.camera_pitch) * (baseline[1]) / depth).item())

        return torch.roll(image, shifts=(disparity_map_y, disparity_map_x), dims=(-2, -1))
    else:
        _, _, H, W = image.size()
        # Roll the image
        disparity_map_y = (param.focal_length/param.camera_pitch) * (baseline[0]) / depth_map
        disparity_map_x = (param.focal_length/param.camera_pitch) * (baseline[1]) / depth_map

        # 그리드 생성
        xx = torch.arange(0, W).view(1, -1).float().repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).float().repeat(1, W)
        grid = torch.stack((xx, yy), dim=0).to(image.device)

        # disparity_map을 이용해 그리드를 조정합니다.
        grid[0, :, :] = (grid[0, :, :] + disparity_map_x) % (W)  # x 축 roll 효과
        grid[1, :, :] = (grid[1, :, :] + disparity_map_y) % (H)  # y 축 roll 효과

        # 그리드를 [-1, 1] 범위로 정규화
        grid[0, :, :] = 2.0 * grid[0, :, :] / (W - 1) - 1.0
        grid[1, :, :] = 2.0 * grid[1, :, :] / (H - 1) - 1.0

        # grid_sample을 사용하여 img1을 샘플링합니다.
        grid = grid.permute(1, 2, 0).unsqueeze(0)  # grid shape를 (N, H, W, 2)로 조정
        shifted_img = F.grid_sample(image, grid, mode='bilinear', padding_mode='reflection', align_corners=False)

        return shifted_img

def trapez(y,y0,w):
    return np.clip(np.minimum(y+1+w/2-y0, -y+1+w/2+y0),0,1)

def metric2pixel(metric, depth, args):
    return int(metric * args.param.focal_length / (depth * args.param.equiv_camera_pitch))

# Data Structure Management

class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value