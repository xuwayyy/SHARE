import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import spectral.io.envi as envi


def get_cube(hdrfile):
    img = envi.open(hdrfile)
    info = envi.read_envi_header(hdrfile)
    if "reflectance scale factor" in info:
        img = img.asarray() / float(info["reflectance scale factor"])
    else:
        img = img.asarray()

    wavelengths = [float(v) for v in info['wavelength']]
    return img, wavelengths


def load_and_crop_Agri(path: Path, target_size=320, band_division=3):
    cube, wavelengths = get_cube(path)
    print(f"[Info] Original cube shape (H, W, C): {cube.shape}, dtype: {cube.dtype}")
    print(f"[Info] Original wavelengths count: {len(wavelengths)}")

    h, w, c = cube.shape
    center_h, center_w = h // 2, w // 2
    half_size = target_size // 2

    cropped_cube = cube[
                   center_h - half_size: center_h + half_size,
                   center_w - half_size: center_w + half_size,
                   ::band_division,
                   ]
    cropped_wavelengths = wavelengths[::band_division]

    print(f"[Info] Cropped cube shape: {cropped_cube.shape}, Cropped wavelengths count: {len(cropped_wavelengths)}")
    return cropped_cube, cropped_wavelengths


class AgriFoodDataset(Dataset):
    def __init__(self, patch: np.ndarray, wavelengths: list, transform=None, retain_ratio=1.0):
        super(AgriFoodDataset, self).__init__()
        self.transform = transform

        if len(patch.shape) == 3 and patch.shape[0] < patch.shape[-1]:
            patch = np.transpose(patch, (1, 2, 0))

        in_ch_full = patch.shape[-1]
        in_ch = int(round(in_ch_full * retain_ratio))

        patch = patch[:, :, :in_ch]
        self.wavelengths = np.array(wavelengths[:in_ch])  #

        patch_min = np.min(patch)
        patch_max = np.max(patch)
        self.patch = (patch - patch_min) / (patch_max - patch_min + 1e-8)
        self.patch = self.patch.astype(np.float32)

        assert transform is not None, 'transform must be defined'

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.transform:
            tensor_patch = self.transform(self.patch)
        else:
            tensor_patch = torch.FloatTensor(self.patch).permute(2, 0, 1)
        return tensor_patch


def makeAgriFoodDataloader(mat_path, transform, patch_size: int, retain_ratio: float = 1.0,
                           band_division: int = 3):
    patch, wavelengths = load_and_crop_Agri(path=mat_path, target_size=patch_size, band_division=band_division)
    dataset = AgriFoodDataset(patch=patch, wavelengths=wavelengths, transform=transform, retain_ratio=retain_ratio)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    return dataloader, dataset.wavelengths


if __name__ == '__main__':
    mat_path = Path( '/mnt/backup/zyx/xjw/Share/data/AgriFood/UseCase_1_(Avoine1)_Anomaly_Easy_L13_6.bil.hdr')

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataloader, wavelengths = makeAgriFoodDataloader(
        mat_path=mat_path,
        transform=transform,
        patch_size=480,
        retain_ratio=1.0,
        band_division=4
    )

    for data in dataloader:
        hsi = data[0].numpy()
        print(f"[Info] HSI tensor shape for network input: {hsi.shape}")

        print(
            f"[Info] Using actual wavelengths, range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm, count: {len(wavelengths)}")

        B_mask = (wavelengths >= 530) & (wavelengths <= 560)
        G_mask = (wavelengths >= 540) & (wavelengths <= 590)
        R_mask = (wavelengths >= 585) & (wavelengths <= 725)

        if not np.any(B_mask):
            print("[Warning] No bands found for B range (530-560nm). Check wavelength coverage.")
        if not np.any(G_mask):
            print("[Warning] No bands found for G range (540-590nm). Check wavelength coverage.")
        if not np.any(R_mask):
            print("[Warning] No bands found for R range (585-725nm). Check wavelength coverage.")

        print(
            f"[Info] B bands indices: {np.where(B_mask)[0]}, range: {wavelengths[B_mask][0] if np.any(B_mask) else 'N/A'}-{wavelengths[B_mask][-1] if np.any(B_mask) else 'N/A'} nm")
        print(
            f"[Info] G bands indices: {np.where(G_mask)[0]}, range: {wavelengths[G_mask][0] if np.any(G_mask) else 'N/A'}-{wavelengths[G_mask][-1] if np.any(G_mask) else 'N/A'} nm")
        print(
            f"[Info] R bands indices: {np.where(R_mask)[0]}, range: {wavelengths[R_mask][0] if np.any(R_mask) else 'N/A'}-{wavelengths[R_mask][-1] if np.any(R_mask) else 'N/A'} nm")

        B = hsi[B_mask].mean(axis=0) if np.any(B_mask) else np.zeros_like(hsi[0])
        G = hsi[G_mask].mean(axis=0) if np.any(G_mask) else np.zeros_like(hsi[0])
        R = hsi[R_mask].mean(axis=0) if np.any(R_mask) else np.zeros_like(hsi[0])


        rgb = np.stack([R, G, B], axis=-1).astype(np.float32)

        rgb_uint8 = np.zeros_like(rgb, dtype=np.uint8)

        for c in range(3):
            channel = rgb[:, :, c]

            p_low = np.percentile(channel, 2)
            p_high = np.percentile(channel, 98)

            if p_high - p_low < 1e-6:
                channel_norm = np.zeros_like(channel)
            else:
                channel_norm = (channel - p_low) / (p_high - p_low)

            channel_norm = np.clip(channel_norm, 0.0, 1.0)
            rgb_uint8[:, :, c] = (channel_norm * 255).astype(np.uint8)

        save_path = '/mnt/backup/zyx/xjw/Share/data/AgriFood/pseudo_RGB.png'
        Image.fromarray(rgb_uint8, mode='RGB').save(save_path)
        print(f"[Success] Saved pseudo-RGB visualization to: {save_path}")

        break