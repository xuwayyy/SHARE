import torch
import deepinv as dinv
from deepinv.physics import GaussianNoise, PoissonNoise, PoissonGaussianNoise
from deepinv.physics import Downsampling, Inpainting, Denoising
from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
import h5py
from typing import Tuple


def get_physics(task, device, img_size: Tuple, factor=None, filter='gaussian', sigma=0.1, mat_index=None,
                        noise_type='gaussian', gain=1/40, filter_params=None):
    if noise_type == 'gaussian':
        noise = GaussianNoise(sigma=sigma)
    elif noise_type == 'poisson':
        noise = PoissonNoise(gain=gain, clip_positive=True)
    elif noise_type == 'gaussian_poisson':
        noise = PoissonGaussianNoise(gain=gain, sigma=sigma)
    else:
        raise ValueError(f'noise_type {noise_type} not recognized')

    if task in ["sr",  "test_sr"]:
        if factor is None:
            raise ValueError('Run SR Experiments But No Factor')
        if img_size is None:
            raise ValueError('Run SR Experiments But No Image Size, Provide Tuple with Image Size[C,H,W]')
        downsample = Downsampling(img_size=img_size, factor=factor, filter=filter, device=device, filter_params=filter_params)
        physics = downsample
        physics.noise_model = noise
    elif task == 'inpainting' or task == "test_inpainting":
        assert mat_index in [1, 2, 3, 4], 'Provide a Mask Index from [1, 2, 3, 4]'
        image_size = 144
        data_dict = loadmat(r'./data/Matzoo/mask_144_144_type_{}.mat'.format(mat_index))
        mask = data_dict['mask']  # 36 36 1 1
        mask = np.array(mask, order='F')
        single_mask = torch.Tensor(mask).to(device)
        single_mask = single_mask.reshape((1, image_size, image_size))
        inpainting = Inpainting(mask=single_mask, tensor_size=img_size, device=device)
        physics = inpainting
        physics.noise_model = noise
    else:
        raise NotImplementedError(f'{task} has not implemented yet in physics yet')
    return physics


