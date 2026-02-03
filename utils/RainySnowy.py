import numpy as np 
import os 
import torch
import pado 
from pado.optical_element import *
from utils.utils import *

class RainySnowy():
    """Generate obstruction of rainy and snowy day."""
    def __init__(self, args, depths):
        """
        Args:
            depth_level: How many depth levels 
            rainrate: rainrate in millimeters per hour
        """
        self.args = args 
        self.param = args.param 
        self.num_depth_level = self.param.num_depth_level
        self.rainrate = self.param.rainrate 
        self.Vfov = 2*np.arctan(self.param.camera_resolution[0]/(2*self.param.focal_length/self.param.camera_pitch)) # vertical FOV in radian
        self.Hfov = 2*np.arctan(self.param.camera_resolution[1]/(2*self.param.focal_length/self.param.camera_pitch)) # horizontal FOV in radian
        self.depths = depths 

        self.rain_alpha = 0.255 # empirically set value
        self.snow_alpha = None # to be implemented

    def marshall_parlmer_dist(self, D):
        """
        https://en.wikipedia.org/wiki/Raindrop_size_distribution
        Marshall-Parlmer distribution. Dimension of return value: m^-3 * mm^-1.
        When return value is multiplied by some diameter range value d(mm), it becomes 
        the number of raindrops in unit volume (1m^3), whose diameter range is D-d/2 ~ D+d/2.
        Args:
            D: diameter of raindrop in mm.
        """ 
        N0 = 8000 
        Lambda = 4.1*(self.rainrate**(-0.21))
        return N0 * np.exp(-Lambda*D)

    def compute_num_raindrop_in_depth_plane(self, depth, diameter_range):
        """
        Args:
            depth: How many depth levels 
            diameter_range: rainrate in millimeters per hour
        Return:
            numbers of raindrops of each diameter and according diameters""" 
        volume_y = 2*np.tan(self.Vfov/2)*depth
        volume_x = 2*np.tan(self.Hfov/2)*depth
        diameters = np.linspace(0, 2, np.round(2//diameter_range).astype(np.int32)) # unit is mm, in the graph of marshall_parlmer dist value diminishes after 2mm
        num_densities = self.marshall_parlmer_dist(diameters) * diameter_range 
        volume_z = (self.param.depth_far_max - self.param.depth_near_min) / self.num_depth_level
        num_raindrops = num_densities * volume_x * volume_y * volume_z
        return num_raindrops, diameters
    
    def compute_rainy_snowy(self, image_far):
        """Generate raindrop obstruction"""
        def _clip_coordinates(x, y, img_w, img_h):
            x = max(0, min(x, img_w))
            y = max(0, min(y, img_h))
            return (x, y)

        def _draw_raindrop(image_alpha_channel, mask, positions, raindrop_size):
            h, w = image_alpha_channel.shape[-2], image_alpha_channel.shape[-1]
            raindrop_h, raindrop_w = raindrop_size
            if raindrop_height_pixel > 1 and raindrop_width_pixel > 1:
                for position in positions:
                    start_x, start_y = _clip_coordinates(position[0], position[1], w, h)
                    end_x, end_y = _clip_coordinates(position[0]+raindrop_w, position[1]+raindrop_h, w, h)
                    start_x = round(start_x)
                    start_y = round(start_y)
                    end_x = round(end_x)
                    end_y = round(end_y)                
                    image_alpha_channel[..., start_y:end_y, start_x:end_x] += self.rain_alpha
                    mask[..., start_y:end_y, start_x:end_x] = 1.0
            return image_alpha_channel, mask

        h, w = image_far.shape[-2], image_far.shape[-1]
        masks = list()
        image_alpha_channels = list()
        image_alpha_channel_sum = np.zeros((h, w))

        for depth in self.depths:
            num_raindrops, diameters = self.compute_num_raindrop_in_depth_plane(depth, 0.1)
            mask_per_depth = np.zeros((h, w))
            image_alpha_channel_per_depth = np.zeros((h,w))
            for num_raindrop, diameter in zip(num_raindrops, diameters): # diameter is in millimeter
                num_raindrop = np.round(num_raindrop).astype(np.int32)
                x_positions = np.random.randint(0, w, size=num_raindrop)
                y_positions = np.random.randint(0, h, size=num_raindrop)
                positions = np.vstack((x_positions, y_positions)).T
                # object pixel size = (size_in_meter * focal_length) / (object_depth * sensor_pitch) 
                raindrop_width_pixel = (diameter/1e+3*self.param.focal_length)/(depth* self.param.camera_pitch * self.param.image_sample_ratio)
                raindrop_height_meter = self.param.shutter_speed * 9 # terminal velocity of raindrop is usually 9
                raindrop_height_pixel = (raindrop_height_meter*self.param.focal_length)/(depth* self.param.camera_pitch * self.param.image_sample_ratio)
                image_alpha_channel_per_depth, mask_per_depth = _draw_raindrop(image_alpha_channel_per_depth, mask_per_depth, positions, (raindrop_height_pixel, raindrop_width_pixel))
            masks.append(mask_per_depth)
            image_alpha_channels.append(image_alpha_channel_per_depth)
            image_alpha_channel_sum = image_alpha_channel_sum + image_alpha_channel_per_depth
        # far depth from near depth
        masks.reverse()
        self.masks = masks 
        self.image_alpha_channels = image_alpha_channels
        self.image_alpha_channel_sum = image_alpha_channel_sum
        return self.image_alpha_channel_sum, masks 
    
    def evaluate_doe(self, DOE_phase, image_far:torch.tensor=None, phase_error=False, chromatic_aberration=False, 
                     obstruction_reinforcement=False, sensor_noise=False, combination=False,
                     image_dirt:torch.tensor=None, dirt_mask:torch.tensor=None):
        """ 
        Args:
            combination: if True, use dirt or other obstruction together with weather (rain, snow) obstruction
            image_dirt: Dirt obstruction near image
        """
        if image_far is None:
            image_far = torch.tensor(np.load(os.path.join(self.args.eval_path, 'img_far_'+str(self.param.img_res)+'.npy')), device=self.args.device).detach()
        DOE_train = DOE(dim=(1, 1, self.param.R,self.param.C), 
                    pitch=self.param.DOE_pitch, 
                    material=self.param.material, 
                    wvl=self.param.DOE_wvl, device=self.args.device)
        
        if phase_error is True:
            phase_noise = torch.rand(DOE_phase.shape) * 2 * self.param.DOE_phase_noise_scale - self.param.DOE_phase_noise_scale
            DOE_phase_with_noise = torch.clamp(DOE_phase + phase_noise.type_as(DOE_phase), -np.pi, np.pi)
            DOE_train.set_phase_change(DOE_phase_with_noise)
        else:
            DOE_train.set_phase_change(DOE_phase)

        psf_far, _ = plot_depth_based_psf(DOE_train, self.args, depths = [5.0], wvls='design', 
                                            normalize = False, merge_channel = True, pad=True)
        convolved_far = fft_convolve2d(image_far.get_intensity() if isinstance(image_far, pado.light.Light) else image_far, 
                                            psf_far)
        convolved_near_sum = torch.zeros_like(convolved_far)
        convolved_mask_sum = torch.zeros_like(convolved_far)
        for idx, depth in enumerate(np.flip(np.linspace(0.1, 5.0, self.param.num_depth_level))):
            # from far depth to near depth
            psf_near, _ = plot_depth_based_psf(DOE_train, self.args, depths = [depth], 
                                                wvls='RGB' if chromatic_aberration else 'design', 
                                                normalize = False, merge_channel = True, pad=True)
            
            mask = torch.tensor(self.masks[idx][np.newaxis, np.newaxis, ...], device=self.args.device)
            image_near = torch.tensor(self.image_alpha_channels[idx][np.newaxis, np.newaxis, ...], device=self.args.device)
            if chromatic_aberration:
                mask = torch.concat([mask, mask, mask], axis=-3)
                image_near = torch.concat([image_near, image_near, image_near], axis=-3)
            convolved_near = fft_convolve2d(image_near, psf_near)
            convolved_mask = fft_convolve2d(mask, psf_near)
            convolved_near_sum = convolved_near_sum+convolved_near
            convolved_mask_sum = convolved_mask_sum+convolved_mask

        # Obstruction reinforcement for rain
        if obstruction_reinforcement is True:
            convolved_near_sum = torch.clamp(1.5*convolved_near_sum, 0, 1)

        convolved_mask_sum = torch.clip(convolved_mask_sum, 0, 1)
        img_conv = convolved_far + convolved_near_sum * convolved_mask_sum

        if sensor_noise is True:
            noise = torch.rand(img_conv.shape) * 2 * self.args.sensor_noise - self.args.sensor_noise
            img_conv = torch.clamp(img_conv + noise.type_as(img_conv), 0, 1)
            convolved_far = torch.clamp(convolved_far + noise.type_as(img_conv), 0, 1)

        if combination:
            if image_dirt is None:
                image_dirt = torch.tensor(np.load(os.path.join(self.args.eval_path, 'img_near_'+str(self.args.param.img_res)+'.npy')), device=self.args.device).detach()
            if dirt_mask is None:
                dirt_mask = torch.tensor(np.load(os.path.join(self.args.eval_path, 'mask_'+str(self.args.param.img_res)+'.npy')), device=self.args.device).detach()

            psf_near, _ = plot_depth_based_psf(DOE_train, self.args, depths = [self.param.depth_near_min], 
                                            wvls='RGB' if chromatic_aberration is True else 'design', 
                                            normalize = False, merge_channel = True, pad=True)

            convolved_far = fft_convolve2d(image_far.get_intensity() if isinstance(image_far, pado.light.Light) else image_far, 
                                            psf_far)
            convolved_near = fft_convolve2d(image_dirt, psf_near)
            convolved_mask = fft_convolve2d(dirt_mask, psf_near)

            # Obstruction reinforcement for dirt
            if obstruction_reinforcement is True:
                convolved_mask = torch.clamp(1.5*convolved_mask, 0, 1)
            
            img_conv = img_conv * (1 - convolved_mask) + convolved_near * convolved_mask
    
        return  (image_far[0].permute(1, 2, 0).cpu().detach().numpy(),
                img_conv[0].permute(1, 2, 0).cpu().detach().numpy(),
                convolved_far[0].permute(1, 2, 0).cpu().detach().numpy(),
                convolved_near_sum, convolved_mask_sum)
