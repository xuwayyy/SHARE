import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path


def read_tivita_hsi(path, normalization: int | None = None) -> np.ndarray:
    """读取 TIVITA HSI .dat 文件"""
    path = Path(path)
    assert path.exists() and path.is_file(), f"Data cube {path} does not exist or is not a file"

    # 读取形状信息 (3个 int32)
    shape = np.fromfile(path, dtype=">i", count=3)

    # 读取数据立方体 (float32, 大端序)
    cube = np.fromfile(path, dtype=">f", offset=12)
    cube = cube.reshape(*shape)

    # 调整空间维度方向以匹配常规图像坐标系
    cube = np.flip(cube, axis=1)
    cube = np.swapaxes(cube, 0, 1)
    cube = cube.astype(np.float32)

    if normalization is not None:
        cube = cube / np.linalg.norm(cube, ord=normalization, axis=2, keepdims=True)
        cube = np.nan_to_num(cube, copy=False)

    return cube


def load_and_crop_HeiPor(path, target_size=320) -> np.ndarray:
    """加载并中心裁剪 HSI 数据，返回 (H, W, C) 格式"""
    cube = read_tivita_hsi(path, normalization=None)  # 暂不归一化，保留原始相对比例
    print(f"[Info] Original cube shape (H, W, C): {cube.shape}, dtype: {cube.dtype}")

    h, w, c = cube.shape
    center_h, center_w = h // 2, w // 2
    half_size = target_size // 2

    # 中心裁剪
    cropped = cube[
              center_h - half_size: center_h + half_size,
              center_w - half_size: center_w + half_size,
              :
              ]
    return cropped  # 保持 (H, W, C) 格式


class HeiPorDataset(Dataset):
    def __init__(self, patch: np.ndarray, transform=None, retain_ratio=1.0):
        super(HeiPorDataset, self).__init__()
        self.transform = transform

        # 1. 统一确保输入为 (H, W, C) 格式
        if patch.shape[0] == 100 and len(patch.shape) == 3:  # 如果误传了 (C, H, W)
            patch = np.transpose(patch, (1, 2, 0))

        assert patch.shape[-1] == 100, f"Expected 100 channels, got {patch.shape[-1]}"

        # 2. 根据 retain_ratio 截取通道 (保持 H, W, C)
        in_ch_full = patch.shape[-1]
        in_ch = int(round(in_ch_full * retain_ratio))
        patch = patch[:, :, :in_ch]  # (H, W, in_ch)

        # 3. 【关键】全局归一化到 [0, 1]，为网络训练提供稳定的输入分布
        # 同时这也为后续可视化保留了真实的通道间相对亮度比例
        patch_min = np.min(patch)
        patch_max = np.max(patch)
        self.patch = (patch - patch_min) / (patch_max - patch_min + 1e-8)
        self.patch = self.patch.astype(np.float32)

        assert transform is not None, 'transform must be defined'

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # self.patch 形状为 (H, W, C)，值域 [0, 1]
        if self.transform:
            # ToTensor 会自动将 (H, W, C) 的 numpy 数组转换为 (C, H, W) 的 Tensor
            # 且因为输入已经是 float32 [0,1]，它不会错误地除以 255
            tensor_patch = self.transform(self.patch)
        else:
            # 手动转换维度
            tensor_patch = torch.FloatTensor(self.patch).permute(2, 0, 1)

        return tensor_patch


def makeHeiPorDataloader(mat_path, transform, patch_size: int, retain_ratio: float = 1.0):
    patch = load_and_crop_HeiPor(path=mat_path, target_size=patch_size)
    dataset = HeiPorDataset(patch=patch, transform=transform, retain_ratio=retain_ratio)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    return dataloader


if __name__ == '__main__':
    mat_path = Path(
        '/mnt/backup/zyx/xjw/Share/data/HeiPorSPECTRAL/'
        '2021_04_28_08_49_12_SpecCube.dat'
    )
    # mat_path = Path(
    #     '/mnt/backup/zyx/xjw/Share/data/HeiPorSPECTRAL/'
    #     '2021_04_15_09_22_02_SpecCube.dat'
    # )

    # 定义 Transform：将 (H, W, C) [0,1] numpy 数组转为 (C, H, W) [0,1] Tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataloader = makeHeiPorDataloader(
        mat_path=mat_path,
        transform=transform,
        patch_size=480,
        retain_ratio=1.0
    )

    for data in dataloader:
        # data shape: [B, C, H, W] -> 取第一个样本得到 [C, H, W]
        hsi = data[0].numpy()
        print(f"[Info] HSI tensor shape for network input: {hsi.shape}")  # 应为 (100, 320, 320)

        # --------------------------------
        # 1. 生成精确的波长数组 (修复了原 np.linspace 的边界偏移问题)
        # --------------------------------
        # 100 bands, 500-995 nm, step 5 nm
        wavelengths = np.arange(500, 1000, 5)
        assert len(wavelengths) == hsi.shape[0], "Wavelength array length mismatch!"

        # --------------------------------
        # 2. 根据指定波长范围选择 bands
        # --------------------------------
        B_mask = (wavelengths >= 530) & (wavelengths <= 560)
        G_mask = (wavelengths >= 540) & (wavelengths <= 590)
        R_mask = (wavelengths >= 585) & (wavelengths <= 725)

        print(
            f"[Info] B bands indices: {np.where(B_mask)[0]}, range: {wavelengths[B_mask][0]}-{wavelengths[B_mask][-1]} nm")
        print(
            f"[Info] G bands indices: {np.where(G_mask)[0]}, range: {wavelengths[G_mask][0]}-{wavelengths[G_mask][-1]} nm")
        print(
            f"[Info] R bands indices: {np.where(R_mask)[0]}, range: {wavelengths[R_mask][0]}-{wavelengths[R_mask][-1]} nm")

        # --------------------------------
        # 3. 对每个波长范围进行平均
        # 注意：此时 hsi 的值域是 [0, 1]，求平均后 R, G, B 的值域依然是 [0, 1]
        # 且它们之间保留了真实的相对辐射强度比例
        # --------------------------------
        B = hsi[B_mask].mean(axis=0)
        G = hsi[G_mask].mean(axis=0)
        R = hsi[R_mask].mean(axis=0)

        # --------------------------------
        # 4. 堆叠并保存 RGB (不再对单通道做 Min-Max 归一化)
        # --------------------------------
        rgb = np.stack([R, G, B], axis=-1)  # shape: (H, W, 3), values in [0, 1]

        # 【方法一：全局 Min-Max 归一化】
        # 简单有效，能将图像最暗处映射为0，最亮处映射为255，且保持RGB通道间的相对比例
        # rgb_min = np.min(rgb)
        # rgb_max = np.max(rgb)
        # rgb_normalized = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)

        # 【方法二：百分位拉伸 (Percentile Stretch)】🌟 高光谱最推荐
        # 高光谱数据常有坏点或噪声极值，直接用 max 会被拉偏。
        # 使用 1% 和 99% 分位数进行拉伸，能有效抑制噪声，让主体更亮、对比度更好。
        # 如果想用这个方法，请注释掉上面的方法一，取消下面两行的注释：
        p_low, p_high = np.percentile(rgb, (1, 99))
        rgb_normalized = np.clip((rgb - p_low) / (p_high - p_low + 1e-8), 0.0, 1.0)

        # 【方法三：Gamma 校正 (可选辅助)】
        # 如果拉伸后仍然觉得暗部细节不够，可以加上 Gamma 校正。
        # gamma < 1.0 会提亮暗部，gamma > 1.0 会压暗暗部。常用 0.45 ~ 0.6
        # rgb_normalized = np.power(rgb_normalized, 0.5)

        # 直接映射到 [0, 255] 并裁剪保存
        # rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        rgb_uint8 = np.clip(rgb_normalized * 255.0, 0, 255).astype(np.uint8)

        save_path = (
            '/mnt/backup/zyx/xjw/Share/data/HeiPorSPECTRAL/'
            'pseudo_RGB.png'
        )

        Image.fromarray(rgb_uint8, mode='RGB').save(save_path)
        print(f"[Success] Saved pseudo-RGB visualization to: {save_path}")

        break  # 只有一个样本，处理完即可退出