import os
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch


def load_and_crop_PaviaUni(path, target_size=256):
    mat = loadmat(path)
    key = [k for k in mat.keys() if not k.startswith('__')][0]
    mat = np.array(mat[key], dtype=np.float32)
    h, w, _ = mat.shape
    center_h, center_w = h // 2, w // 2
    half_size = target_size // 2

    cropped = mat[
              center_h - half_size: center_h + half_size,
              center_w - half_size: center_w + half_size,
              :
              ]
    cropped = (cropped - np.min(cropped)) / (np.max(cropped) - np.min(cropped))

    return cropped


class PaviaUniDataset(Dataset):
    def __init__(self, patch, transform=None, retain_ratio=1.0):
        super(PaviaUniDataset, self).__init__()
        self.transform = transform

        if patch.shape[-1] == 103:
            patch = np.transpose(patch, [2, 0, 1])

        in_ch_full = patch.shape[0]  # 103
        in_ch = int(round(in_ch_full * retain_ratio))
        self.patch = patch[:in_ch, :, :]  # (in_ch, H, W)

        assert transform is not None, 'transform must be defined in PaviaCenterDataset'

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        patch = self.patch  # (C, H, W) numpy array

        if self.transform:
            patch = np.transpose(patch, (1, 2, 0))  # (C,H,W) → (H,W,C)
            patch = self.transform(patch)  # ToTensor: (H,W,C) → (C,H,W) tensor
        else:
            patch = torch.FloatTensor(patch)

        return patch


def makePaviaDataLoader(mat_path, transform, patch_size, retain_ratio=1.0):
    patch = load_and_crop_PaviaUni(path=mat_path, target_size=patch_size)
    dataset = PaviaUniDataset(patch=patch, transform=transform, retain_ratio=retain_ratio)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    return dataloader


def loadPaviaLabelWithinPatch(mat_path, patch_size):
    mat = loadmat(mat_path)
    key = [k for k in mat.keys() if not k.startswith('__')][0]
    mat = np.array(mat[key], dtype=np.uint8)
    h, w = mat.shape
    center_h, center_w = h // 2, w // 2
    half_size = patch_size // 2

    cropped = mat[
              center_h - half_size: center_h + half_size,
              center_w - half_size: center_w + half_size,
              ]
    print(f"original shape: {mat.shape}, cropped shape: {cropped.shape}")
    print(f"cropped categories: {np.unique(cropped)}")
    print(f"non-cropped categories: {np.unique(mat)}")

    return cropped


