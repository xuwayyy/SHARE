import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch


def load_and_crop_Chikusei(path, target_size=256, offset=(0, 0)):
    with h5py.File(path, 'r') as f:
        key = list(f.keys())[0]
        mat = np.array(f[key], dtype=np.float32)


    _, h, w = mat.shape
    center_h, center_w = h // 2, w // 2

    offset_h, offset_w = offset
    new_center_h = center_h - offset_h
    new_center_w = center_w - offset_w

    half_size = target_size // 2

    start_h = max(new_center_h - half_size, 0)
    end_h = start_h + target_size
    start_w = max(new_center_w - half_size, 0)
    end_w = start_w + target_size

    if end_h > h:
        start_h = h - target_size
        end_h = h
    if end_w > w:
        start_w = w - target_size
        end_w = w

    cropped = mat[:, start_h:end_h, start_w:end_w]
    cropped = (cropped - np.min(cropped)) / (np.max(cropped) - np.min(cropped))

    return cropped



class ChikuseiDataset(Dataset):
    def __init__(self, patch, transform=None):
        super(ChikuseiDataset, self).__init__()
        self.transform = transform
        self.patch = patch
        if self.patch.shape[0] != 128:
            self.patch = np.transpose(self.patch, [2, 0, 1])
        assert transform is not None, 'transform must be defined in PaviaCenterDataset'

    def __len__(self):
        # return len(self.patch)
        return 1

    def __getitem__(self, idx):
        return self.patch


def makeChikuseiDataLoader(mat_path, transform, patch_size, offset=(120, 120)):
    patch = load_and_crop_Chikusei(path=mat_path, target_size=patch_size, offset=offset)
    dataset = ChikuseiDataset(patch=patch, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    return dataloader


if __name__ == '__main__':
    load_and_crop_Chikusei(path="")
