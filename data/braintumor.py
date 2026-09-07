import os
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ================= 配置路径 =================
hdr_path1 = "/mnt/backup/zyx/xjw/Share/data/ThirdCampaignraw/042-01SecondCampaign.hdr"
hdr_path2 = "/mnt/backup/zyx/xjw/Share/data/ThirdCampaignraw/043-01SecondCampaign.hdr"
output_rgb_dir = "/mnt/backup/zyx/xjw/Share/data/ThirdCampaignraw/Visualizations"
os.makedirs(output_rgb_dir, exist_ok=True)


# ================= 1. 解析 Header =================
def parse_envi_header(hdr_path):
    with open(hdr_path, 'r', encoding='utf-8') as f:
        header_content = f.read()

    samples = int(re.search(r'samples\s*=\s*(\d+)', header_content, re.I).group(1))
    lines = int(re.search(r'lines\s*=\s*(\d+)', header_content, re.I).group(1))
    bands = int(re.search(r'bands\s*=\s*(\d+)', header_content, re.I).group(1))
    data_type = int(re.search(r'data type\s*=\s*(\d+)', header_content, re.I).group(1))
    interleave = re.search(r'interleave\s*=\s*(\w+)', header_content, re.I).group(1).strip().lower()
    byte_order = int(re.search(r'byte order\s*=\s*(\d+)', header_content, re.I).group(1))

    db_match = re.search(r'default bands\s*=\s*\{([^}]+)\}', header_content, re.I | re.DOTALL)
    if db_match:
        # 提取数字并转为 0-based 索引
        default_bands = [int(x) - 1 for x in re.findall(r'\d+', db_match.group(1))]
    else:
        default_bands = [0, min(1, bands - 1), min(2, bands - 1)]

    return {
        'samples': samples, 'lines': lines, 'bands': bands,
        'data_type': data_type, 'interleave': interleave,
        'byte_order': byte_order, 'default_bands': default_bands
    }


# ================= 2. 读取二进制数据 =================
def load_hsi_data(hdr_path):
    """根据 header 信息读取对应的二进制数据文件，并 reshape 为 (H, W, C)"""
    meta = parse_envi_header(hdr_path)
    base_name = os.path.splitext(hdr_path)[0]

    # 寻找实际的数据文件
    data_path = None
    for ext in ['.HSI', '.hsi', '.img', '.dat', '.raw', '']:
        if os.path.exists(base_name + ext):
            data_path = base_name + ext
            break

    if data_path is None:
        raise FileNotFoundError(f"找不到 {base_name} 对应的数据文件")

    # 映射 ENVI data type 到 numpy dtype
    if meta['data_type'] == 12:
        dtype = np.uint16
    elif meta['data_type'] == 4:
        dtype = np.float32
    else:
        dtype = np.uint16

    endian = '<' if meta['byte_order'] == 0 else '>'
    np_dtype = np.dtype(dtype).newbyteorder(endian)

    # 读取数据
    data = np.fromfile(data_path, dtype=np_dtype)

    # 根据 interleave 方式 reshape 为 (H, W, C)
    if meta['interleave'] == 'bil':
        data = data.reshape((meta['lines'], meta['bands'], meta['samples']))
        data = np.transpose(data, (0, 2, 1))  # (lines, samples, bands)
    elif meta['interleave'] == 'bsq':
        data = data.reshape((meta['bands'], meta['lines'], meta['samples']))
        data = np.transpose(data, (1, 2, 0))
    elif meta['interleave'] == 'bip':
        data = data.reshape((meta['lines'], meta['samples'], meta['bands']))
    else:
        raise ValueError(f"不支持的 interleave 类型: {meta['interleave']}")

    return data, meta


# ================= 3. 裁剪与降采样 =================
def load_and_crop_Brain(hdr_path, target_size=320, band_division=10):
    """加载、中心裁剪并波段降采样 HSI 数据，返回 (H, W, C) 格式"""
    data, meta = load_hsi_data(hdr_path)
    h, w, c = data.shape

    center_h, center_w = h // 2, w // 2
    half_size = target_size // 2

    # 边界保护，防止 target_size 大于原图尺寸
    start_h = max(0, center_h - half_size)
    end_h = min(h, center_h + half_size)
    start_w = max(0, center_w - half_size)
    end_w = min(w, center_w + half_size)

    # 空间裁剪 + 波段降采样
    cropped = data[start_h:end_h, start_w:end_w, ::band_division]
    print(f"[Info] Original shape: ({h}, {w}, {c}) -> Cropped & Decimated shape: {cropped.shape}")

    return cropped, meta


# ================= 4. 伪 RGB 可视化 =================
def visualize_pseudo_rgb(cropped_data, orig_default_bands, band_division, save_path):
    """从降采样后的数据中提取伪 RGB 并保存"""
    # 原始推荐波段 (0-based) 例如 [108, 191, 424] 对应 [蓝, 绿, 红]
    # 我们需要按 [红, 绿, 蓝] 顺序提取，所以反转
    target_orig_indices = sorted(orig_default_bands[:3], reverse=True)

    # 计算降采样后的近似索引，并确保不越界
    max_c = cropped_data.shape[-1]
    target_indices = [min(idx // band_division, max_c - 1) for idx in target_orig_indices]
    print(f"[Info] Extracting RGB from decimated band indices: {target_indices} (Original: {target_orig_indices})")

    # 提取 R, G, B 通道
    rgb_data = cropped_data[:, :, target_indices].astype(np.float32)

    # 2% ~ 98% 对比度拉伸 (忽略极暗和手术灯反光极亮值)
    p2 = np.percentile(rgb_data, 2, axis=(0, 1), keepdims=True)
    p98 = np.percentile(rgb_data, 98, axis=(0, 1), keepdims=True)
    denominator = np.maximum(p98 - p2, 1e-6)

    rgb_normalized = (rgb_data - p2) / denominator * 255.0
    rgb_normalized = np.clip(rgb_normalized, 0, 255).astype(np.uint8)

    # 保存图像
    img = Image.fromarray(rgb_normalized)
    img.save(save_path)
    print(f"[Info] Pseudo-RGB image saved to: {save_path}")


# ================= 5. PyTorch Dataset =================
class BrainDataset(Dataset):
    def __init__(self, patch: np.ndarray, transform=None, retain_ratio=1.0):
        super(BrainDataset, self).__init__()
        self.transform = transform

        assert len(patch.shape) == 3, f"Expected 3D array (H, W, C), got shape {patch.shape}"
        h, w, c = patch.shape

        # 根据 retain_ratio 截取通道 (保持 H, W, C)
        in_ch = max(1, int(round(c * retain_ratio)))
        patch = patch[:, :, :in_ch]

        # 全局归一化到 [0, 1]，为网络训练提供稳定的输入分布
        patch_min = np.min(patch)
        patch_max = np.max(patch)
        self.patch = (patch - patch_min) / (patch_max - patch_min + 1e-8)
        self.patch = self.patch.astype(np.float32)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.transform:
            # ToTensor 会自动将 (H, W, C) 的 ndarray 转换为 (C, H, W) 的 Tensor
            # 因为 self.patch 已经是 float32 且在 [0, 1] 之间，ToTensor 不会再次除以 255
            tensor_patch = self.transform(self.patch)
        else:
            tensor_patch = torch.FloatTensor(self.patch).permute(2, 0, 1)
        return tensor_patch


# ================= 6. DataLoader 构造器 =================
def makeBrainDataloader(hdr_path, transform, patch_size: int, retain_ratio: float = 1.0, band_division=10):
    patch, meta = load_and_crop_Brain(hdr_path=hdr_path, target_size=patch_size, band_division=band_division)

    # 可视化切块后的伪 RGB 图像
    base_name = os.path.basename(hdr_path).replace('.hdr', '')
    vis_path = os.path.join(output_rgb_dir, f"{base_name}_cropped_rgb.png")
    visualize_pseudo_rgb(patch, meta['default_bands'], band_division, vis_path)

    dataset = BrainDataset(patch=patch, transform=transform, retain_ratio=retain_ratio)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    return dataloader


# ================= 主执行逻辑 =================
if __name__ == '__main__':
    # 定义 Transform：将 (H, W, C) 转为 (C, H, W) 的 Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print("\n--- 处理第一张图像 ---")
    dataloader1 = makeBrainDataloader(
        hdr_path=hdr_path1,
        transform=transform,
        patch_size=480,
        band_division=10,
        retain_ratio=1.0
    )

    for batch in dataloader1:
        print(f"Batch shape: {batch.shape}, dtype: {batch.dtype}")
        # batch shape 应该是: torch.Size([1, C, H, W])
        break

    print("\n--- 处理第二张图像 ---")
    dataloader2 = makeBrainDataloader(
        hdr_path=hdr_path2,
        transform=transform,
        patch_size=480,
        band_division=10,
        retain_ratio=1  # 演示 retain_ratio 的作用
    )

    for batch in dataloader2:
        print(f"Batch shape: {batch.shape}, dtype: {batch.dtype}")
        break

    print("\n🎉 所有处理完成！请查看可视化输出目录。")


# # ================= 配置路径 =================
# input_dir = r"D:\科研工作\SHARE\data\ThirdCampaignraw"
# output_dir = r"D:\科研工作\SHARE\data\ThirdCampaignrawRGB"
#
# os.makedirs(output_dir, exist_ok=True)
#
#
# # ================= 辅助函数 =================
# def parse_envi_header(hdr_path):
#     """解析 ENVI .hdr 文件，提取关键元数据"""
#     with open(hdr_path, 'r', encoding='utf-8') as f:
#         header_content = f.read()
#
#     samples = int(re.search(r'samples\s*=\s*(\d+)', header_content, re.I).group(1))
#     lines = int(re.search(r'lines\s*=\s*(\d+)', header_content, re.I).group(1))
#     bands = int(re.search(r'bands\s*=\s*(\d+)', header_content, re.I).group(1))
#     data_type = int(re.search(r'data type\s*=\s*(\d+)', header_content, re.I).group(1))
#     interleave = re.search(r'interleave\s*=\s*(\w+)', header_content, re.I).group(1).strip().lower()
#     byte_order = int(re.search(r'byte order\s*=\s*(\d+)', header_content, re.I).group(1))
#
#     db_match = re.search(r'default bands\s*=\s*\{([^}]+)\}', header_content, re.I | re.DOTALL)
#     if db_match:
#         # 提取数字并转为 0-based 索引
#         default_bands = [int(x) - 1 for x in re.findall(r'\d+', db_match.group(1))]
#     else:
#         default_bands = [0, min(1, bands - 1), min(2, bands - 1)]
#
#     return {
#         'samples': samples, 'lines': lines, 'bands': bands,
#         'data_type': data_type, 'interleave': interleave,
#         'byte_order': byte_order, 'default_bands': default_bands
#     }
#
#
# def get_data_type_numpy(data_type_code, byte_order):
#     if data_type_code == 12:
#         dtype = np.uint16
#     elif data_type_code == 4:
#         dtype = np.float32
#     elif data_type_code == 1:
#         dtype = np.uint8
#     else:
#         dtype = np.uint16
#     endian = '<' if byte_order == 0 else '>'
#     return np.dtype(dtype).newbyteorder(endian)
#
#
# def process_hsi_to_rgb(hdr_path, output_dir):
#     try:
#         meta = parse_envi_header(hdr_path)
#         base_name = os.path.splitext(hdr_path)[0]
#
#         data_path = None
#         for ext in ['.HSI', '.hsi', '.img', '.dat', '.raw', '']:
#             if os.path.exists(base_name + ext):
#                 data_path = base_name + ext
#                 break
#
#         if data_path is None:
#             print(f"⚠️ 警告: 找不到 {base_name} 对应的数据文件，跳过。")
#             return
#
#         dtype = get_data_type_numpy(meta['data_type'], meta['byte_order'])
#         data = np.fromfile(data_path, dtype=dtype)
#
#         if meta['interleave'] == 'bil':
#             data = data.reshape((meta['lines'], meta['bands'], meta['samples']))
#             data = np.transpose(data, (0, 2, 1))
#         elif meta['interleave'] == 'bsq':
#             data = data.reshape((meta['bands'], meta['lines'], meta['samples']))
#             data = np.transpose(data, (1, 2, 0))
#         elif meta['interleave'] == 'bip':
#             data = data.reshape((meta['lines'], meta['samples'], meta['bands']))
#         else:
#             raise ValueError(f"不支持的 interleave 类型: {meta['interleave']}")
#
#         # ================= 核心 1：纠正 RGB 映射顺序 =================
#         # 将推荐的波段按波长从大到小排序，确保：长波->R, 中波->G, 短波->B
#         bands_to_extract = sorted(meta['default_bands'][:3], reverse=True)
#         bands_to_extract = [min(b, meta['bands'] - 1) for b in bands_to_extract]
#         rgb_data = data[:, :, bands_to_extract]
#
#         # ================= 核心 2：按通道进行 2%~98% 对比度拉伸 =================
#         rgb_float = rgb_data.astype(np.float32)
#
#         # 分别计算 R, G, B 三个通道的 2% 和 98% 分位数 (忽略极暗和极亮反光)
#         p2 = np.percentile(rgb_float, 2, axis=(0, 1), keepdims=True)
#         p98 = np.percentile(rgb_float, 98, axis=(0, 1), keepdims=True)
#
#         # 防止分母为 0 (如果某个通道全是0)
#         denominator = np.maximum(p98 - p2, 1e-6)
#
#         # 线性映射到 0-255
#         rgb_normalized = (rgb_float - p2) / denominator * 255.0
#
#         # 截断超出范围的值并转为 8-bit 无符号整数
#         rgb_normalized = np.clip(rgb_normalized, 0, 255).astype(np.uint8)
#
#         output_filename = os.path.basename(base_name) + ".png"
#         output_path = os.path.join(output_dir, output_filename)
#
#         img = Image.fromarray(rgb_normalized)
#         img.save(output_path)
#         print(f"✅ 成功处理: {os.path.basename(hdr_path)} -> {output_filename}")
#
#     except Exception as e:
#         print(f"❌ 处理 {hdr_path} 时出错: {e}")
#
#
# # ================= 主执行逻辑 =================
# if __name__ == "__main__":
#     print(f"开始扫描目录: {input_dir}")
#     hdr_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.hdr')]
#
#     if not hdr_files:
#         print("未找到任何 .hdr 文件，请检查输入路径。")
#     else:
#         print(f"共找到 {len(hdr_files)} 个 .hdr 文件，开始处理...\n")
#         for hdr_file in hdr_files:
#             hdr_path = os.path.join(input_dir, hdr_file)
#             process_hsi_to_rgb(hdr_path, output_dir)
#
#     print("\n🎉 所有文件处理完成！")