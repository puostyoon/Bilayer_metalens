import numpy as np
from pado.material import *
from utils.utils import *

sensor_dist = focal_length = 4e-3
aperture_shape='circle'

# METASURFACE specs
meta_wvl = 532e-9 # wavelength used to set initial fresnel phase map
meta_pitch = 350e-9 # about 300nm
R = C = 10000 # Resolution of the simulated wavefront
img_res = 1024

# Training parameters
phase_init = torch.randn((1,1, R, C))
aperture_diamter = meta_pitch * R 
full_broadband_wvls = np.linspace(400, 700, 301)*1e-9
training_wvls = np.linspace(480, 620, 141)*1e-9 # 450, 550, 635
camera_pitch = 1.0e-6  # 1.5e-6
