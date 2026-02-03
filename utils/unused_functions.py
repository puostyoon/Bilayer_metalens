
def create_2d_phase_map(phase_vectors_vertical, phase_vectors_horizontal, phase_scales, args):
    """
    Create 2D phase map based on Singular Value Decomposition
    """
    R = phase_vectors_vertical.shape[1]
    phase_map = torch.zeros((R, R), device=args.device)
    # phase_vectors_vertical: (rank, R)
    phase_vectors_vertical = phase_vectors_vertical / (phase_vectors_vertical.norm(dim=1, keepdim=True)+1e-20) # normalize
    phase_vectors_horizontal = phase_vectors_horizontal / (phase_vectors_horizontal.norm(dim=1, keepdim=True)+1e-20) # normalize
    phase_scales = phase_scales.unsqueeze(1) # (rank) -> (rank, 1)
    # (scale[0] * v[0].T @ h[0] + scale[1] * v[1].T @ h[1] + ... + scale[rank-1] * v[rank-1].T @ h[rank-1]
    phase_map = (phase_vectors_vertical.T @ (phase_scales*phase_vectors_horizontal)) # (R, rank) @ (rank, R) -> (R, R)

    # (R, R) -> (1, 1, R, R)
    phase_map = phase_map.unsqueeze(0).unsqueeze(0)
    return phase_map

def create_2d_circular_phase_map(phase_vector, diameter):
    """
    Create a 2D phase map by rotating a 1D phase vector into a circular pattern.

    Args:
    - phase_vector (torch.Tensor): 1D phase vector.
    - diameter (int): The diameter of the phase map.

    Returns:
    - torch.Tensor: Rotated 2D phase map.
    """
    N = phase_vector.shape[0]
    R = diameter // 2
    phase_map = torch.zeros((diameter, diameter), device=phase_vector.device)

    # Creating a grid of indices
    y, x = torch.meshgrid(torch.arange(diameter), torch.arange(diameter), indexing='ij')
    x = x - R
    y = y - R
    distance = torch.sqrt(x**2 + y**2).float()

    indices = torch.round(distance).long()
    valid = (indices < N) & (distance <= R)  

    phase_map[valid] = phase_vector[indices[valid]]

    # (2400, 2400) -> (1, 1, 2400, 2400)
    phase_map = phase_map.unsqueeze(0).unsqueeze(0)

    return phase_map

def create_1d_parametrization(DOE):
    """
    Convert a 2D phase map to a 1D parametrization by averaging over concentric circles.

    Args:
    - phase_map (torch.Tensor): A 2D tensor of shape (1, 1, R, C) representing the phase map.

    Returns:
    - torch.Tensor: A 1D tensor representing the radial average of the phase map.
    """
    # Ensure the phase map is square and calculate the center

    phase_map = DOE.get_phase_change()

    assert phase_map.shape[2] == phase_map.shape[3], "Phase map must be square."
    R = phase_map.shape[2]
    center = R // 2

    # Create a grid of coordinates centered at the middle
    y, x = torch.meshgrid(torch.arange(R), torch.arange(R), indexing='ij')
    x = x - center
    y = y - center

    # Calculate the radius for each point in the grid
    radius = torch.sqrt(x**2 + y**2)

    # Calculate the maximum radius to consider (slightly inside the image to avoid boundary issues)
    max_radius = center

    # Prepare the 1D tensor to store the averages
    radial_profile = torch.zeros(max_radius, device=phase_map.device)

    # Iterate over each possible radius and compute the mean
    for r in range(max_radius):
        mask = (radius >= r) & (radius < r+1)
        if mask.any():
            radial_profile[r] = phase_map[:, :, mask].mean()

    DOE_phase = create_2d_circular_phase_map(radial_profile, R)
    DOE.set_phase_change(DOE_phase)

    return DOE

def psf_thin_lens_far(args, z_far = 5, offset=None, normalize=True):

    param = args.param
    prop = Propagator('SBL_ASM')
    dim = (1,1,param.R, param.C)
    
    light = Light(dim, param.DOE_pitch, param.DOE_wvl, device=args.device)
    light.set_spherical_light(np.array([z_far]))


    lens = pado.optical_element.RefractiveLens(dim, param.DOE_pitch, param.focal_length, param.DOE_wvl, args.device)
    light = lens.forward(light.clone())
    aperture = Aperture(dim, param.DOE_pitch, param.aperture_diamter, param.aperture_shape, param.DOE_wvl, args.device)
    light = aperture.forward(light.clone())
    light_prop = prop.forward(light, param.focal_length, offset)
    psf_far = light_prop.get_intensity()

    if args.resizing_method == 'original':
        psf_far = F.interpolate(psf_far, scale_factor=light_prop.pitch / param.DOE_pitch)
        psf_far = sample_psf(psf_far, param.DOE_sample_ratio)
    else:
        psf_far = F.interpolate(psf_far, scale_factor=light_prop.pitch/(param.camera_pitch* param.image_sample_ratio), 
                            mode=args.resizing_method)
           
    if normalize:
        psf_far = psf_far / torch.sum(psf_far)

    return psf_far

def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf):
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1-c0) < abs(r1-r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)

    # The following is now always < 1 in abs
    slope = (r1-r0) / (c1-c0)

    # Adjust weight by the slope
    w *= np.sqrt(1+np.abs(slope)) / 2

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1+1, dtype=float)
    y = x * slope + (c1*r0-c0*r1) / (c1-c0)

    # Now instead of 2 values for y, we have 2*np.ceil(w/2).
    # All values are 1 except the upmost and bottommost.
    thickness = np.ceil(w/2)
    yy = (np.floor(y).reshape(-1,1) + np.arange(-thickness-1,thickness+2).reshape(1,-1))
    xx = np.repeat(x, yy.shape[1])
    vals = trapez(yy, y.reshape(-1,1), w).flatten()

    yy = yy.flatten()

    # Exclude useless parts and those outside of the interval
    # to avoid parts outside of the picture
    mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))

    return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])

def backward_image_loss_3_channel(args, batch_data, DOE_phase, wvls, thetas, net=None):
    """
    Image restoration loss without using the reconstruction network. 
    """
    param = args.param
    image_far, _ = batch_data
    image_far = image_far.to(args.device)

    for (theta_y, theta_x) in thetas:
        img_conv_3channel = list()
        for channel_idx, wvl in enumerate(wvls):
            DOE_train = DOE((1,1,param.R, param.R), param.DOE_pitch, param.material, wvl=param.DOE_wvl, device=args.device)
            DOE_train.set_phase_change(DOE_phase + (torch.randn((DOE_phase.shape), device=args.device) * args.phase_noise_stddev).detach() if args.phase_noise_stddev is not None else DOE_phase) 

            image_far_one_channel, _, mask, convolved_far, _, _, img_conv, psf_near, psf_far = image_formation(image_far,
                                                                                DOE_train, 
                                                                                args.compute_obstruction, 
                                                                                args, 
                                                                                channel_idx=(None if len(wvls)==1 else channel_idx),
                                                                                wvls=wvl,
                                                                                theta=(theta_y, theta_x),
                                                                                pad=True,
                                                                                psf_shape = 'horizontal')
            
            if args.image_noise_stddev is not None:
                img_conv = img_conv + (torch.randn((img_conv.shape), device=args.device) * args.image_noise_stddev).detach()

            # Adjust contrast
            scale_factor_contrast = torch.clamp(image_far_one_channel.std()/img_conv.std(), max=args.contrast_clamp)
            print('Contrast scale factor:', scale_factor_contrast)
            img_conv = img_conv.mean() + scale_factor_contrast * (img_conv - img_conv.mean())
            img_conv_3channel.append(img_conv)

        img_conv = torch.cat(img_conv_3channel, dim=-3)
        brightness_regularizer = (1-image_far.mean()/convolved_far.mean())**2

        if args.use_network:
            img_conv = net(img_conv)

        img_conv_mask = img_conv * mask
        image_far_mask = image_far * mask
        l1_loss = args.l1_loss_weight * args.l1_criterion(img_conv, image_far)
        print('l1 loss: ', l1_loss, 'brightness regularizer: ', brightness_regularizer)
        masked_loss = args.masked_loss_weight * args.l1_loss_weight * args.l1_criterion(img_conv_mask, image_far_mask)
        loss = l1_loss + masked_loss + brightness_regularizer * args.brightness_regularizer_coeff

        if args.use_perc_loss:
            perc_loss = torch.mean(args.perceptual_loss_weight * 
                                args.perceptual_criterion(2 * img_conv.to(torch.float32) - 1, 2 * image_far.to(torch.float32) - 1))
            loss = loss + perc_loss

        if args.use_da_loss:
            # debugging
            da_loss = DA_loss(img_conv, image_far, args, feature='segmentation')
            loss = loss + da_loss
            # debugging

        loss = loss * args.image_loss_weight
        loss.backward()

    return loss.item()