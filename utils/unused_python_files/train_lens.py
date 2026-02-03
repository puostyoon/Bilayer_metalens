import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import lpips

import argparse
from tqdm.auto import tqdm
import os
import sys
import time
from importlib.machinery import SourceFileLoader
import numpy as np

from utils.utils import *
from utils.Dirt import *
from utils.ROLE import compute_raindrop
from utils.Defence import compute_fence 
from models.DA_loss_functions import DA_loss

def lens_image_formation(image_far, DOE, compute_obstruction, args, channel_idx=None, z_near = None, wvls='RGB', theta=(0, 0), pad=True, far_disparity=False, concat=False):
    param = args.param

    z_far = randuni(param.depth_far_min, param.depth_far_max, 1)[0] # randomly sample the far-point depth from a range
    image_far = image_far.to(torch.float32)
    psf_far, _ = plot_depth_based_psf(DOE, args, depths = [z_far], wvls=wvls, merge_channel = True, pad=pad, theta=theta, normalize=False)

    if args.normalizing_method == 'new' and (psf_far.shape[-2] > param.camera_resolution[-2] and psf_far.shape[-1] > param.camera_resolution[-1]):
        normalizer = psf_far[...,psf_far.shape[-2]//2-param.camera_resolution[-2]//2:psf_far.shape[-2]//2+param.camera_resolution[-2]//2,
                            psf_far.shape[-1]//2-param.camera_resolution[-1]//2:psf_far.shape[-1]//2+param.camera_resolution[-1]//2].sum()
    else:
        normalizer = psf_far.sum() if psf_far.shape[-3]==1 else psf_far[0,1,...].sum()

    psf_far = psf_far/normalizer

    if channel_idx is not None:
        image_far = image_far[:, channel_idx:channel_idx+1, ...]
        image_near = image_near[:, channel_idx:channel_idx+1, ...]

    convolved_far = fft_convolve2d(image_far.get_intensity() if isinstance(image_far, pado.light.Light) else image_far, 
                                   psf_far)
    
    return convolved_far, image_far, normalizer

def backward_image_loss(args, batch_data, DOE_phase, wvls, thetas, net=None):
    """
    Image restoration loss without using the reconstruction network. 
    """
    param = args.param
    for (theta_y, theta_x) in thetas:
        step_loss = 0
        for wvl in wvls:
            # compute loss weight for broadband
            if args.train_broadband: # loss weight for each wvl
                loss_weight = 0
                for color_ in ['R', 'G', 'B']:
                    loss_weight = loss_weight + param.cam_response_func[wvl][color_] * param.DOE_eff[wvl] * param.bandpass_filter_transmission[wvl]
                if loss_weight < 10: # To speed up the training
                    continue
                print('wvl: ',wvl, 'loss_weight: ', loss_weight)

            DOE_train = DOE((1,1,param.R, param.R), param.DOE_pitch, param.material, wvl=param.DOE_wvl, device=args.device)
            DOE_train.set_phase_change(DOE_phase + (torch.randn((DOE_phase.shape), device=args.device) * args.phase_noise_stddev).detach() if args.phase_noise_stddev is not None else DOE_phase) 

            image_far_color, _ = batch_data
            image_far_color = image_far_color.to(args.device)

            # Since channel_idx is None, we convolve single channel psf with 3-channel color image
            convolved_far, image_far, normalizer = lens_image_formation(image_far_color,
                                                                        DOE_train, 
                                                                        args.compute_obstruction, 
                                                                        args, 
                                                                        channel_idx=None,
                                                                        wvls=wvl,
                                                                        theta=(theta_y, theta_x),
                                                                        pad=True,
                                                                        concat=False,
                                                                        far_disparity=False)
            
            if args.image_noise_stddev is not None:
                convolved_far = convolved_far + (torch.randn((convolved_far.shape), device=args.device) * args.image_noise_stddev).detach()

            # Adjust contrast
            scale_factor_contrast = torch.clamp(image_far.std()/convolved_far.std(), max=args.contrast_clamp)
            print('Contrast scale factor:', scale_factor_contrast)
            convolved_far = convolved_far.mean() + scale_factor_contrast * (convolved_far - convolved_far.mean())
            incident_light_intensity_sum = np.pi*((param.R/2)**2)*((param.DOE_pitch/param.camera_pitch)**2) # four psf are summed
            brightness_regularizer = (incident_light_intensity_sum/normalizer-1)
            
            if args.use_network:
                convolved_far = net(convolved_far)

            loss = args.l1_loss_weight * args.l1_criterion(convolved_far, image_far)
            print('l1 loss: ', loss, 'brightness regularizer: ', brightness_regularizer)
            if args.use_perc_loss:
                perc_loss = torch.mean(args.perceptual_loss_weight * 
                                    args.perceptual_criterion(2 * convolved_far.to(torch.float32) - 1, 2 * image_far.to(torch.float32) - 1))
                
                loss = loss + perc_loss

            if args.use_da_loss:
                da_loss = DA_loss(convolved_far, image_far, args, feature='segmentation')
                loss = loss + da_loss 

            loss = loss * args.image_loss_weight + brightness_regularizer * args.brightness_regularizer_coeff

            if args.train_broadband: # loss weight for each wvl
                loss = loss * loss_weight / param.loss_weight_normalizer
            
            loss.backward()
            step_loss = step_loss + loss.item()

    return step_loss

def train(args):

    writer = SummaryWriter(log_dir=args.result_path + '/runs')
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    param = args.param

    transform_train = transforms.Compose([
            transforms.RandomCrop(param.data_resolution,pad_if_needed=True), # Places365 image size varies
            transforms.RandomCrop([param.equiv_crop_size, param.equiv_crop_size],pad_if_needed=True),
            transforms.Resize([param.img_res, param.img_res]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
            transforms.Resize([param.img_res, param.img_res]),
            transforms.ToTensor(),
        ])
    if args.obstruction == 'fence':
        trainset = torchvision.datasets.Places365(
            root=param.dataset_dir, split="train-standard", transform=transform_train)
        testset = torchvision.datasets.Places365(
            root=param.dataset_dir, split="val", transform=transform_test)   
        args.compute_obstruction = compute_fence     
    elif args.obstruction == 'raindrop':
        trainset = torchvision.datasets.ImageFolder(param.training_dir, transform=transform_train)
        testset = torchvision.datasets.ImageFolder(param.val_dir, transform=transform_test)
        args.compute_obstruction = compute_raindrop
    elif args.obstruction == 'dirt':
        trainset = torchvision.datasets.ImageFolder(param.training_dir, transform=transform_train)
        testset = torchvision.datasets.ImageFolder(param.val_dir, transform=transform_test)
        args.compute_obstruction = compute_dirt
    elif args.obstruction == 'dirt_raindrop':
        trainset = torchvision.datasets.ImageFolder(param.training_dir, transform=transform_train)
        testset = torchvision.datasets.ImageFolder(param.val_dir, transform=transform_test)
        args.compute_obstruction = compute_dirt_raindrop
    elif args.obstruction == 'dirt_fence':
        trainset = torchvision.datasets.ImageFolder(param.training_dir, transform=transform_train)
        testset = torchvision.datasets.ImageFolder(param.val_dir, transform=transform_test)
        args.compute_obstruction = compute_dirt_fence

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # build model and loss
    args.l1_criterion = nn.L1Loss().to(args.device)
    
    if args.use_perc_loss:
        args.perceptual_criterion = lpips.LPIPS(net='vgg').to(args.device)

    DOE_phase = torch.tensor(param.DOE_phase_init.to(args.device), requires_grad=True)
    optics_optimizer = optim.Adam([DOE_phase], lr=args.optics_lr)

    if args.use_network:
        # from models.RefineNetwork import RefineNetwork
        # net = RefineNetwork(args).to(args.device)
        from models.localnet import ParamLocal
        # from models.localnet_4imgs import ParamLocal
        net = ParamLocal(args).to(args.device)
        if args.pretrained_network is not None:
            net.load_state_dict(torch.load(args.pretrained_network, map_location='cpu'))
            net.to(args.device)
        net_optimizer = optim.Adam(params=net.parameters(), lr=args.network_lr)
    else:
        net = None 
        net_optimizer = None 

    total_step = 0
    eval_minimum_loss = np.inf

    wvls = param.wvls if args.train_RGB else (param.broadband_wvls if args.train_broadband else [param.DOE_wvl])
    if args.train_broadband:
        loss_weight_normalizer = 0
        for wvl_ in wvls:
            for color_ in ['R', 'G', 'B']:
                loss_weight_normalizer += param.cam_response_func[wvl_][color_] * param.DOE_eff[wvl_] * param.bandpass_filter_transmission[wvl_]
        param.loss_weight_normalizer = loss_weight_normalizer

    thetas = [(0, 0)]
    if args.spatially_varying_PSF:
        theta_xs = np.round(np.clip(abs(np.random.normal(0, param.FOV//4, 3)), 0, param.FOV//2))
        theta_ys = np.round(np.clip(abs(np.random.normal(0, param.FOV//4, 3)), 0, param.FOV//2))
        for y, x in zip(theta_ys, theta_xs):
            thetas.append((y, x))
            thetas.append((y, -x))
            thetas.append((-y, x))
            thetas.append((-y, -x))
    for epoch in tqdm(range(args.n_epochs), position=0, leave=False):
        for step, batch_data in enumerate(tqdm(trainloader)):
            image_loss = backward_image_loss(args, batch_data, DOE_phase, wvls, thetas, net)
            optics_optimizer.step()
            optics_optimizer.zero_grad()
            if args.use_network:
                net_optimizer.step()
                net_optimizer.zero_grad()

            with torch.no_grad():
                if total_step%args.log_freq==0 and total_step>0:
                    eval_loss = 0
                    eval_loss = evaluate_image_loss(args, DOE_phase, net, testloader, writer, total_step, single_sensor=False)
                    # Save the best model
                    if eval_loss < eval_minimum_loss:
                        eval_minimum_loss = eval_loss
                        torch.save(DOE_phase, os.path.join(args.result_path,f'DOE_phase_minimum_eval_loss.pt'))
                        if args.use_network:
                            torch.save(net.state_dict(), os.path.join(args.result_path,f'net_minimum_eval_loss.pt'))

            
            total_step += 1
            os.system('clear')
            print(f'\n Epoch {epoch} step {step} Loss (image): {image_loss}')
            print('\n DOE phase shape: ', DOE_phase.shape)

            # save model
            if total_step % args.save_freq == 0:
                torch.save(DOE_phase, os.path.join(args.result_path,'DOE_phase_%03d.pt' % (total_step//args.save_freq)))
                if args.use_network:
                    torch.save(net.state_dict(), os.path.join(args.result_path,'network_%03d.pt' % (total_step//args.save_freq)))

            print("optics optimizer lr: ", optics_optimizer.param_groups[0]['lr'])

    writer.close()  


def main():
    parser = argparse.ArgumentParser(
        description='PSF based Obstruction-free Metasurface training',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    def none_or_str(value):
        if value == 'None':
            return None
        return value
    
    parser.add_argument('--debug', action="store_true", help='debug mode, train on validation data to speed up the process')
    parser.add_argument('--result_path', default = './example/asset/ckpt/metasurface', type=str, help='dir to save models and checkpoints')
    parser.add_argument('--eval_path', default="dataset/eval", type=str, help='path where an image for evaluation is saved')
    parser.add_argument('--param_file', default= './example/asset/config/param_MV_1600_metasurface.py', type=str, help='path to param file')
    parser.add_argument('--pretrained_DOE', default = None, type=none_or_str, help = 'Directory of pretrained DOE or None')
    parser.add_argument('--pretrained_network', default = None, type=none_or_str, help = 'Directory of pretrained network or None')

    # Related to dataset usage (used for image restoration loss)
    parser.add_argument('--use_dataset', default = False, action="store_true", help='Use dataset for doe training. Used when using image restoration loss')
    parser.add_argument('--use_network', default = False, action="store_true", help='Use refinement network for end-to-end training')
    parser.add_argument('--fence_dataset_dir', help='Directory of dataset used for fence obstruction')
    parser.add_argument('--dirt_raindrop_dataset_train_dir', help='Directory of dataset used for raindrop, dirt, and raindrop_dirt obstructions')
    parser.add_argument('--dirt_raindrop_dataset_val_dir', help='Directory of dataset used for raindrop, dirt, and raindrop_dirt obstructions')

    parser.add_argument('--obstruction', default = 'dirt_raindrop', type = str, help = 'obsturction type')
    parser.add_argument('--propagator', default = 'Fraunhofer', type = str, help = 'propagator used to compute the psf')
    parser.add_argument('--use_lens', action="store_true", help = 'Additional lens usage. Look at compute_psf_arbitrary_prop. Note that Fresnel+Lens is Fraunhofer')
    parser.add_argument('--n_epochs', default = 1, type = int, help = 'max num of training epoch')
    parser.add_argument('--optics_lr', default=0.1, type=float, help='optical element learning rate')
    parser.add_argument('--network_lr', default=1e-4, type=float, help='reconstruction network learning rate')

    # Related to loss
    parser.add_argument('--use_perc_loss', action="store_true", help = 'use lpips perceptual loss')
    parser.add_argument('--use_da_loss', action="store_true", help = 'use domain adaptation loss')
    parser.add_argument('--da_loss_weight', default = 0.1, type = float, help = 'weight for domain adaptation loss')
    parser.add_argument('--l1_loss_weight', default = 1, type = float, help = 'weight for L1 loss')
    parser.add_argument('--masked_loss_weight', default = 1, type = float, help = 'weight for masked loss (focus on obstructed scene)')
    parser.add_argument('--perceptual_loss_weight', default = 1, type = float, help = 'weight for perceptual loss')
    parser.add_argument('--image_loss_weight', default=1.0, type=float, help='weight for image reconstruction loss') 
    parser.add_argument('--psf_loss_weight', default=1.0, type=float, help='weight for psf reconstruction loss') 
    parser.add_argument('--brightness_clamp', default=1.0, type=float, help='Maximum brightness adjustment value before applying loss function')
    parser.add_argument('--brightness_regularizer_coeff', default=0.001, type=float, help='Brightness regularizer coefficient')
    parser.add_argument('--contrast_clamp', default=1.0, type=float, help='Maximum contrast (stddev) adjustment value before applying loss function')

    # Related to training method
    parser.add_argument('--resizing_method', default='area', type=str, help='PSF resizing method. original or area. look at compute_psf function')
    parser.add_argument('--normalizing_method', default='original', type=str, help='PSF normalizing method. If original, use full psf for normalzing. if new, use sensor size for normalizing')
    parser.add_argument('--train_RGB', action="store_true", help='If False, use single wvl psf for all color channel. If True, use color-variant psf.')
    parser.add_argument('--train_broadband', action="store_true", help='If True, train optics and network for broadband wvls')
    parser.add_argument('--constant_wvl_phase', action="store_true", help='If True, do not call DOE.change_wvl() ever.')
    parser.add_argument('--spatially_varying_PSF', action="store_true", help='If False, use single wvl psf for all color channel. If True, use color-variant psf.')
    parser.add_argument('--image_noise_stddev',  default=0.01, type=float, help='Stddev of the gaussian noise added to image before computing the loss.')
    parser.add_argument('--phase_noise_stddev',  default=0.2, type=float, help='Stddev of the gaussian noise added to phase (radian) before propagation.')

    # Related to reconstruction network usage (LocalNet)
    parser.add_argument("--inverse",   action="store_true",  help="Inverse the tone curve")
    parser.add_argument("--no-grad",   action="store_false", dest="use_grad",     help="Disable gradients")
    parser.add_argument("--no-hist",   action="store_false", dest="use_hist",     help="Disable soft histograms")
    parser.add_argument("--no-satmask", action="store_false", dest="use_satmask", help="Disable saturation mask")

    parser.add_argument('--log_freq', default=30, type=int, help = 'frequency (num_steps) of logging')
    parser.add_argument('--save_freq', default=400, type=int, help = 'frequency (num_steps) of saving checkpoint and visual performance')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--phase_init', default='zero', type=str, help='zero, random, fresnel')
    parser.add_argument('--device', default='cpu', type=str, help='torch_device')
    parser.add_argument('--batch_size', default=1, type=int, help='image data batch size')

    args = parser.parse_args()

    param = SourceFileLoader("param", args.param_file).load_module()
    param = convert_resolution(param,args)

    assert not (args.train_RGB and args.train_broadband) # two options are mutually exclusive.
    assert args.brightness_clamp >= 1
    assert args.contrast_clamp >= 1

    if args.pretrained_DOE is not None:
        param.DOE_phase_init = torch.load(args.pretrained_DOE, map_location=args.device).detach()
    else:
        if args.phase_init=='random':
            param.DOE_phase_init = torch.rand(param.DOE_phase_init.shape, device=args.device) * 10
        elif args.phase_init=='fresnel':
            param.DOE_phase_init = RefractiveLens(param.DOE_phase_init.shape, param.DOE_pitch, param.focal_length, param.DOE_wvl, args.device).get_phase_change()
        else:
            # zero initialization
            param.DOE_phase_init = torch.zeros(param.DOE_phase_init.shape, device=args.device)
            print('ze initialized DOE phase')
    
    save_settings(args, param)

    if args.spatially_varying_PSF:
        args.SVPSF_fitting_table = np.load('example/temp_logs/fitting' + ('' if param.focal_length==10e-3 else f'_{param.FOV}') + '.npy') 

    if args.use_dataset:
        # train using image dataset and image reconstruction loss
        train(args)
    else:
        raise NotImplementedError('We do not only use PSF-based loss.')

if __name__ == '__main__':
    main()
