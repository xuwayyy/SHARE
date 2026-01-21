import h5py
import numpy as np
from scipy.io import loadmat
import os
import torch
from data.physics import get_physics


def min_max_norm_channel(x):
    channel = x.shape[-1]
    normalized_channels = []
    for c in range(channel):
        per_channel_img = x[:, :, c]
        min_val = np.min(per_channel_img)
        max_val = np.max(per_channel_img)
        range_val = max_val - min_val

        if range_val == 0:
            normalized_channels.append(np.zeros_like(per_channel_img, dtype=np.float32))
        else:
            normalized_channels.append((per_channel_img - min_val) / range_val)

    normalized_img = np.stack(normalized_channels, axis=-1)
    return normalized_img

def load_inpainting_mat(chikusei_index=2):
    chikusei = h5py.File("./data/Matzoo/Chikusei_Test_5_images.mat")
    chikusei = chikusei["gt_blob"]
    indian_pine = loadmat("./data/Matzoo/Indian_pines_corrected.mat")
    indian_pine = indian_pine["indian_pines_corrected"]

    chikusei = np.array(chikusei, dtype=np.float32, order='F')
    chikusei = chikusei[:, :, :, chikusei_index]
    indian_pine = np.array(indian_pine, dtype=np.float32, order='F')
    indian_pine = indian_pine[:144, : 144, :]
    # normalized_indian_pine = (indian_pine - np.min(indian_pine)) / (np.max(indian_pine) - np.min(indian_pine))
    normalized_indian_pine = min_max_norm_channel(indian_pine)
    return chikusei, normalized_indian_pine


def get_inpainting_dataset(device, chikusei_index=2):
    bands = [128, 200]
    image_size = 144
    chikusei, indian_pine = load_inpainting_mat(chikusei_index=chikusei_index)
    chikusei = torch.Tensor(chikusei).permute(2, 0, 1).view(1, bands[0], image_size, image_size).to(device)
    indian_pine = torch.Tensor(indian_pine).permute(2, 0, 1).view(1, bands[1], image_size, image_size).to(device)
    mat_zoo = {}
    mat_zoo['indian_pine'] = indian_pine
    mat_zoo['chikusei'] = chikusei
    return mat_zoo


if __name__ == '__main__':
    dataset_dict = get_inpainting_dataset(device="cuda:0")
    chikusei, indian_pine = dataset_dict['chikusei'], dataset_dict['indian_pine']
    physics = get_physics(task='inpainting', device='cuda:0', img_size=(128, 144, 144))
    y = physics(chikusei)
    dagger = physics.A_adjoint(y)
    print(f"y.shape: {y.shape}, dagger.shape: {dagger.shape}")
