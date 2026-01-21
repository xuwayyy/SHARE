import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch


def load_and_crop_PaviaUni(path, target_size=256):
    mat = loadmat(path)
    key = [k for k in mat.keys() if not k.startswith('__')][0]
    mat = np.array(mat[key], dtype=np.float32)
    # mat = (mat - np.min(mat)) / (np.max(mat) - np.min(mat))
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
    def __init__(self, patch, transform=None):
        super(PaviaUniDataset, self).__init__()
        self.transform = transform
        self.patch = patch
        if self.patch.shape[0] != 103:
            self.patch = np.transpose(self.patch, [2, 0, 1])
        assert transform is not None, 'transform must be defined in PaviaCenterDataset'

    def __len__(self):
        # return len(self.patch)
        return 1

    def __getitem__(self, idx):
        return self.patch


def makePaviaDataLoader(mat_path, transform, patch_size):
    patch = load_and_crop_PaviaUni(path=mat_path, target_size=patch_size)
    dataset = PaviaUniDataset(patch=patch, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    return dataloader


if __name__ == '__main__':
    load_and_crop_PaviaUni(path=r"C:\Users\xieji\OneDrive\桌面\HSI\Pavia_Uni\PaviaU.mat", target_size=256)
