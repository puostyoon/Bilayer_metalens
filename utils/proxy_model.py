import torch 
import pado
from pado.light import * 
from pado.optical_element import *
from pado.propagator import *
import torch
import torch.fft
import numpy as np
from .utils import *

def phase2duty(phase_map_at_standard_wvl): # (1, 1, R) # 1D phase map이 input으로 들어감.
    phase = torch.remainder(phase_map_at_standard_wvl, 2 * torch.pi)
    phase = phase/(2*torch.pi) # 0-1 로 바꿈?
    p = [-0.1484, 0.6809, 0.2923] # phase2duty_coefficients
    return p[0] * phase ** 2 + p[1] * phase + p[2]

def duty2phase(duty, wvls): # 위 phase2duty 함수에서 구한 duty를 넣어줌
    """
    Args:
        duty (torch.Tensor): 1D tensor of shape [1, 1, R], where each value corresponds to duty cycle at the standard wvl (452nm in the Neural Nano Optics).
        wvls (torch.Tensor): 1D tensor of shape [len(wvls)].
    Returns:
        phase (torch.Tensor): tensor of shape [1, wvl, R].
    """
    wvls_repeated = (wvls)[None, ..., None].repeat([1, 1, len(duty)]) # [1, len(wvls), R]
    duty = torch.clip(duty, min = 0.3, max = 0.82)
    p = [6.051, -0.02033, 2.26, 1.371E-5, -0.002947, 0.797] # duty2phase_coefficients
    lam = wvls_repeated[..., :duty.shape[-1]] # (1, len(target_wvls), duty.shape[-1])? duty.shape[-1]로 잘라주는 이유?? 딱히 없음. 있으나 마나임. 김서연의 코드에서만 작동함. 
    phase = p[0] + p[1]*lam + p[2]*duty + p[3]*lam**2 + p[4]*lam*duty + p[5]*duty**2 # Neural Nano Optics 따라함.
    return phase*2*torch.pi
# 최종 phase map은 예를 다시 한 바퀴 돌려서 쓰는 거임. 아래에 돌리는 함수 있음.

def radial_to_2d_symmetric_map(v, output_r):
    """
    Converts a 1D radial vector into a 2D rotationally symmetric map using 360° rotation.
   
    Args:
        v (torch.Tensor): 1D tensor of shape [1, len(wvls), R], where each value corresponds to intensity at radius r.
       
    Returns:
        map_2d (torch.Tensor): 2D symmetric tensor of shape [1, wvl, 2R, 2R].
    """
    assert output_r%2 != 0 # Output R should be Odd to be rotationally symmetric
 
    # Create coordinate grid from [-R, R-1]
    r_idx = torch.linspace(-output_r//2+1, output_r//2, output_r).to(v.device)
    y, x = torch.meshgrid(r_idx, r_idx, indexing='ij')
 
    # Compute distance from center
    dist = torch.sqrt((x)**2 + (y)**2)
 
    # Clamp radius values to valid indices [0, R]
    dist_index = torch.clamp(dist, max=output_r/2).to(torch.int64)
 
    # Use radial values as lookup
    map_2d = v[0, :, dist_index].unsqueeze(0)
 
    # Mask:
    mask = (dist<=output_r/2)[None, None, :, :]
    return map_2d*mask

def standard_wvl_phase_1d_to_2d(phase_at_standard_wvl, wvls):
    """
    Args:
        phase_at_standard_wvl (torch.Tensor): 1D tensor of shape [1, 1, R], where each value corresponds to duty cycle at the standard wvl (452nm in the Neural Nano Optics).
        wvls (torch.Tensor): 1D tensor of shape [len(wvls)].
    Returns:
        rotated_phase_map (torch.Tensor): rotated tensor of shape [1, wvl, R, R].
    """
    duty = phase2duty(phase_at_standard_wvl)
    phase_at_different_wvl = duty2phase(duty, wvls)
    rotated_phase_map = radial_to_2d_symmetric_map(phase_at_different_wvl, phase_at_different_wvl.shape[-1]*2-1)
    rotated_phase_map = rotated_phase_map % (2*torch.pi) - torch.pi
    return rotated_phase_map

##### 2D proxy model #####
def phase2duty_2d(phase_map_at_standard_wvl): # (1, 1, R, C) # 2D phase map이 input으로 들어감.
    phase = torch.remainder(phase_map_at_standard_wvl, 2 * torch.pi)
    phase = phase/(2*torch.pi) # 0-1 로 바꿈?
    p = [-0.1484, 0.6809, 0.2923] # phase2duty_coefficients
    return p[0] * phase ** 2 + p[1] * phase + p[2]

##### 2D proxy model #####
def phase2duty_2d(phase_map_at_standard_wvl): # (1, 1, R, C) # 2D phase map이 input으로 들어감.
    phase = torch.remainder(phase_map_at_standard_wvl, 2 * torch.pi)
    phase = phase/(2*torch.pi) # 0-1 로 바꿈?
    p = [-0.1484, 0.6809, 0.2923] # phase2duty_coefficients
    return p[0] * phase ** 2 + p[1] * phase + p[2]

def duty2phase_2d(duty, wvls): # 위 phase2duty_2d 함수에서 구한 duty를 넣어줌
    """
    Args:
        duty (torch.Tensor): 2D tensor of shape [1, 1, R, C], where each value corresponds to duty cycle at the standard wvl (452nm in the Neural Nano Optics).
        wvls (torch.Tensor): 1D tensor of shape [len(wvls)].
    Returns:
        phase (torch.Tensor): tensor of shape [1, wvl, R, C].
    """
    wvls_repeated = (wvls)[None, ..., None].repeat([1, 1, len(duty), len(duty)]) # [1, len(wvls), R, C]
    duty = torch.clip(duty, min = 0.3, max = 0.82)
    p = [6.051, -0.02033, 2.26, 1.371E-5, -0.002947, 0.797] # duty2phase_coefficients
    phase = p[0] + p[1]*wvls_repeated + p[2]*duty + p[3]*wvls_repeated**2 + p[4]*wvls_repeated*duty + p[5]*duty**2 # Neural Nano Optics 따라함.
    return phase*2*torch.pi

def standard_wvl_phase_2d_to_2d(phase_map_at_standard_wvl, wvls):
    """
    Args:
        phase_map_at_standard_wvl (torch.Tensor): 2D tensor of shape [1, 1, R, C], where each value corresponds to duty cycle at the standard wvl (452nm in the Neural Nano Optics).
        wvls (torch.Tensor): 1D tensor of shape [len(wvls)].
    Returns:
        rotated_phase_map (torch.Tensor): rotated tensor of shape [1, wvl, R, C].
    """
    duty_2d = phase2duty_2d(phase_map_at_standard_wvl)
    phase_map = duty2phase_2d(duty_2d, wvls)
    phase_map = phase_map % (2*torch.pi) - torch.pi
    return phase_map


# Image Formation and Simulation

def plot_depth_based_psf_proxy_model(doe, args, depths, wvls = 'RGB', merge_channel = False, pad=True, use_lens=None, propagator=None, eval=False, offset=(0, 0), theta=(0, 0), normalize=True):
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
                psf = compute_psf_arbitrary_prop_proxy_model(wvl, 
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
                psf = compute_psf_arbitrary_prop_proxy_model(wvl, 
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

def image_formation_proxy_model(image_far, DOE, compute_obstruction, args, channel_idx=None, z_near = None, wvls='RGB', theta=(0, 0), pad=True, far_disparity=False, concat=False):
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

    psf_far, _ = plot_depth_based_psf_proxy_model(DOE, args, depths = [z_far], wvls=wvls, merge_channel = True, pad=pad, theta=theta, normalize=False)
    psf_near, _ = plot_depth_based_psf_proxy_model(DOE, args, depths = [z_near], wvls=wvls, merge_channel = True, pad=pad, theta=theta, normalize=False)

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


def compute_psf_arbitrary_prop_proxy_model(wvl, depth, doe, args, propagator='Fraunhofer', use_lens=True, offset=(0, 0), variable_offset_indices=None, theta=(0, 0), pad=True, full_psf=False, normalize=True):
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
    doe.set_phase_change(standard_wvl_phase_2d_to_2d(doe.get_phase_change(), wvls=torch.tensor([wvl], device=args.device)))
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

    light_prop = prop.forward(light, param.sensor_dist, offset=offset, variable_offset_indices=variable_offset_indices, linear=False, scale=scale)
    psf = light_prop.get_intensity()

    # resize 
    psf = F.interpolate(psf, scale_factor=light_prop.pitch/(param.camera_pitch* param.image_sample_ratio), 
                        mode=args.resizing_method)    
    if normalize:
        psf = psf / torch.sum(psf)

    return psf
