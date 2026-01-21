import h5py
import numpy as np
from data.physics import get_physics
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F


def crop_sub_img(data: np.array, patch: tuple, stride: tuple):
    C, H, W = data.shape
    patch_c, patch_h, patch_w = patch
    stride_c, stride_h, stride_w = stride

    patches = []
    for c in range(0, C - patch_c + 1, stride_c):
        for i in range(0, H - patch_h + 1, stride_h):
            for j in range(0, W - patch_w + 1, stride_w):
                patch_data = data[c:c + patch_c, i:i + patch_h, j:j + patch_w]
                patches.append(patch_data)

    return np.array(patches)


def createDataset(mat_path, patch=(31, 512, 512), stride=(31, 512, 512), name='fake_and_real_beers_ms.mat'):
    hr_patches = []
    if name != "all":
        for mat_file in sorted(os.listdir(mat_path)):
            # if mat_file.endswith('.mat'):
            if mat_file == name:
                with h5py.File(os.path.join(mat_path, mat_file), 'r') as f:
                    key = list(f.keys())[0]
                    mat_array = np.array(f[key], dtype=np.float32)
                    if mat_array.ndim == 3:
                        patches = crop_sub_img(mat_array, patch, stride)
                        hr_patches.extend(patches)
        hr_patches = np.array(hr_patches)  # HR
    else:
        for mat_file in sorted(os.listdir(mat_path)):
            with h5py.File(os.path.join(mat_path, mat_file), 'r') as f:
                key = list(f.keys())[0]
                mat_array = np.array(f[key], dtype=np.float32)
                if mat_array.ndim == 3:
                    if mat_array.shape[0] != 31:
                        print(f"mat {mat_file} is not 31, it is {mat_array.shape[0]}")
                    patches = crop_sub_img(mat_array, patch, stride)
                    hr_patches.extend(patches)
        hr_patches = np.array(hr_patches)
    return hr_patches


class CaveDataset(Dataset):
    def __init__(self, hr_patches, transform=None, mode='train', task='single'):
        super().__init__()
        self.task = task
        self.whole_patches = hr_patches
        print("total patches:", len(self.whole_patches))
        if mode == 'train':
            self.hr_patches = hr_patches[: int(len(hr_patches) * 0.6875)]  # 22 / 32 images training
        else:
            self.hr_patches = hr_patches[int(len(hr_patches) * 0.6875):]
        self.transform = transform
        assert transform is not None, 'Please provide a transformation function'
        if mode == "train":
            print(f"Training Samples: {len(self.hr_patches)}")
        else:
            print(f"Test Samples: {len(self.hr_patches)}")
        self.mode = mode

    def __len__(self):
        if self.task == "single":
            return len(self.hr_patches)
        else:  # single image sr, only allow batch=1
            return 1  

    def __getitem__(self, idx):
        if self.task == 'single':
            hr_patch = self.whole_patches[0]
        else:
            hr_patch = self.hr_patches[idx]

        hr_patch = np.transpose(hr_patch, (1, 2, 0))

        if self.transform:
            hr_patch = self.transform(hr_patch)

        return hr_patch


def makeDataLoader(mat_path="", task='single', mode='train', transform=None, bs=1, name='fake_and_real_beers_ms.mat'):
    hr_patches = createDataset(mat_path=mat_path, name=name)
    dataset = CaveDataset(hr_patches, transform=transform, mode=mode, task=task)
    if mode == 'train':
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0)
    else:
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)
    return dataloader


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    mat_path = ""
    hr_patches = createDataset(mat_path=mat_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    physics = get_physics(task="sr", device=device, sigma=0.1, img_size=(31, 256, 256), factor=2)

    train_set = CaveDataset(hr_patches, transform=transform, mode='train')
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    for i, hr in enumerate(train_loader):
        hr = hr.to(device)
        lr = physics(hr)
        print("hr shape: {}, lr shape: {}".format(hr.shape, lr.shape))
        break
