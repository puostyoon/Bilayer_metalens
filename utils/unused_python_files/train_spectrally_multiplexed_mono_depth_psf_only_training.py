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
from models.DA_loss_functions import DA_loss

def compute_psf_loss_mask(depth, psf, sharp, delta=3):

    if sharp:
        r_min = delta
        r_max = np.inf
        # r_max = 120
    else:
        # r_min = (-40*depth + 60) + delta
        r_min = (-9*depth + 40) + delta
        r_max = np.inf

    _, _, H, W = psf.shape
    center_y, center_x = H / 2, W / 2
    y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    distance = torch.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)

    mask = ((distance >= r_min)&(distance <= r_max)).float()
    return mask.to(psf.device)

def backward_psf_loss(DOE_phase, depth, wvl, args, sharp, step, eval=False):
    param = args.param
    DOE_train = DOE((1,1,param.R, param.R), param.DOE_pitch, param.material, wvl=wvl, device=args.device)
    DOE_train.set_phase_change(DOE_phase)

    # PSF loss
    psf = compute_psf_arbitrary_prop(wvl=wvl, 
                                        depth=torch.as_tensor(depth), 
                                        doe=DOE_train, 
                                        args=args, 
                                        propagator='SBL_ASM',
                                        use_lens=args.use_lens, 
                                        theta=(0, 0), 
                                        normalize=False)
    loss_mask = compute_psf_loss_mask(depth, psf, sharp=sharp)
    loss = torch.sum(loss_mask * psf / psf.sum())

    # brightness regularize
    incident_light_intensity_sum = np.pi*((param.R/2)**2)*((param.DOE_pitch/param.camera_pitch)**2) # four psf are summed
    brightness_regularizer = (incident_light_intensity_sum/psf.sum()-1)
    loss = loss + brightness_regularizer 


    if eval:
        plt.figure(); plt.imshow(np.clip((psf/psf.sum()).cpu().detach().numpy()[0, 0, :, :], 0, 1)); plt.colorbar()
        plt.savefig(os.path.join(args.result_path, 'logged_psf', f'step_{step}_depth_{depth}_wvl_{wvl}.png'), bbox_inches='tight'); plt.clf(); plt.close()
    else:
        loss.backward()
    return loss.detach().item()

def train(args):

    writer = SummaryWriter(log_dir=args.result_path + '/runs')
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    param = args.param

    # build model and loss
    args.l1_criterion = nn.L1Loss().to(args.device)
    DOE_phase = torch.tensor(param.DOE_phase_init.to(args.device), requires_grad=True)
    optics_optimizer = optim.Adam([DOE_phase], lr=args.optics_lr)

    total_step = 0
    eval_minimum_loss = np.inf

    depth_wvls = [620e-9]
    sharp_wvls = [430e-9]
    # sharp_wvls = [550e-9]
    # depths = [1, 2, 3, 4, 5]
    depths = [0.5, 1, 1.5, 2, 2.5, 3]
    # depths = [0.25, 0.5, 0.75, 1.0, 1.25]

    for step in tqdm(range(args.n_epochs)):
        psf_loss = 0
        for sharp_wvl in sharp_wvls:
            for depth in depths:
                psf_loss = psf_loss + backward_psf_loss(DOE_phase, depth, sharp_wvl, args=args, step=step, sharp=True)
        for depth_wvl in depth_wvls:
            for depth in depths:
                psf_loss = psf_loss + backward_psf_loss(DOE_phase, depth, depth_wvl, args=args, step=step, sharp=False)
        optics_optimizer.step()
        optics_optimizer.zero_grad()

        if total_step%args.log_freq==0 and total_step>0:
            with torch.no_grad():
                os.makedirs(os.path.join(args.result_path, 'logged_psf'), exist_ok=True)        

                eval_loss = 0
                for sharp_wvl in sharp_wvls:
                    for depth in depths:
                        eval_loss = eval_loss + backward_psf_loss(DOE_phase, depth, sharp_wvl, args=args, step=step, sharp=True, eval=True)

                for depth_wvl in depth_wvls:
                    for depth in depths:
                        eval_loss = eval_loss + backward_psf_loss(DOE_phase, depth, depth_wvl, args=args, step=step, sharp=False, eval=True)
                eval_loss = eval_loss / (len(depth_wvls)*len(depths) + len(sharp_wvls)*len(depths))
                
                # Save the best model
                if eval_loss < eval_minimum_loss:
                    eval_minimum_loss = eval_loss
                    torch.save(DOE_phase, os.path.join(args.result_path,f'DOE_phase_minimum_eval_loss.pt'))
        
        total_step += 1
        os.system('clear')
        print(f'\n step {step} Loss (psf): {psf_loss / (len(depth_wvls)*len(depths) + len(sharp_wvls)*len(depths))}')
        print('\n DOE phase shape: ', DOE_phase.shape)

        # save model
        if total_step % args.save_freq == 0:
            torch.save(DOE_phase, os.path.join(args.result_path,'DOE_phase_%03d.pt' % (total_step//args.save_freq)))

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

    parser.add_argument('--propagator', default = 'Fraunhofer', type = str, help = 'propagator used to compute the psf')
    parser.add_argument('--use_lens', action="store_true", help = 'Additional lens usage. Look at compute_psf_arbitrary_prop. Note that Fresnel+Lens is Fraunhofer')
    parser.add_argument('--n_epochs', default = 1, type = int, help = 'max num of training epoch')
    parser.add_argument('--optics_lr', default=0.1, type=float, help='optical element learning rate')
    parser.add_argument('--network_lr', default=1e-4, type=float, help='reconstruction network learning rate')

    # Related to loss
    parser.add_argument('--use_perc_loss', action="store_true", help = 'use lpips perceptual loss')
    parser.add_argument('--use_da_loss', action="store_true", help = 'use domain adaptation loss')
    parser.add_argument('--l1_loss_weight', default = 1, type = float, help = 'weight for L1 loss')
    parser.add_argument('--psf_loss_weight', default=1.0, type=float, help='weight for psf reconstruction loss') 

    # Related to training method
    parser.add_argument('--resizing_method', default='area', type=str, help='PSF resizing method. original or area. look at compute_psf function')
    parser.add_argument('--normalizing_method', default='original', type=str, help='PSF normalizing method. If original, use full psf for normalzing. if new, use sensor size for normalizing')
    parser.add_argument('--constant_wvl_phase', action="store_true", help='If True, do not call DOE.change_wvl() ever.')
    parser.add_argument('--spatially_varying_PSF', action="store_true", help='If True, do not call DOE.change_wvl() ever.')

    parser.add_argument('--log_freq', default=30, type=int, help = 'frequency (num_steps) of logging')
    parser.add_argument('--save_freq', default=400, type=int, help = 'frequency (num_steps) of saving checkpoint and visual performance')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--phase_init', default='zero', type=str, help='zero, random, fresnel')
    parser.add_argument('--device', default='cpu', type=str, help='torch_device')
    parser.add_argument('--batch_size', default=1, type=int, help='image data batch size')

    args = parser.parse_args()

    param = SourceFileLoader("param", args.param_file).load_module()

    if args.phase_init=='random':
        param.DOE_phase_init = torch.rand(param.DOE_phase_init.shape, device=args.device) * 10
    elif args.phase_init=='fresnel':
        param.DOE_phase_init = RefractiveLens(param.DOE_phase_init.shape, param.DOE_pitch, param.focal_length, param.DOE_wvl, args.device).get_phase_change()
    else:
        # zero initialization
        param.DOE_phase_init = torch.zeros(param.DOE_phase_init.shape, device=args.device)
        print('ze initialized DOE phase')
    
    # save settings
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    else:
        raise Exception("The directory already exists!") 
    args_dict = vars(args)
    with open(os.path.join(args.result_path,'args.json'), "w") as f:
        json.dump(args_dict, f, indent=4, sort_keys=False)
    shutil.copy(args.param_file, args.result_path)
    args.param = param

    train(args)

if __name__ == '__main__':
    main()
