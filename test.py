import torch
from metric import Metric
from matplotlib import pyplot as plt
import numpy as np
from deepinv.physics import Denoising, GaussianNoise
import torch.nn.functional as F
import os
from scipy.io import savemat


def _save_img(path, img, cmap='gray', vmin=None, vmax=None):
    plt.imsave(path, img, cmap=cmap, vmin=vmin, vmax=vmax)


def _to_uint8(tensor, stretch_percentiles=(2, 98), per_channel_stretch=True):
    arr = tensor.squeeze().detach().cpu().numpy()  # [C, H, W]

    if per_channel_stretch:
        p_low = np.percentile(arr, stretch_percentiles[0], axis=(1, 2), keepdims=True)
        p_high = np.percentile(arr, stretch_percentiles[1], axis=(1, 2), keepdims=True)
    else:
        p_low, p_high = np.percentile(arr, stretch_percentiles)
        p_low = np.full((arr.shape[0], 1, 1), p_low, dtype=arr.dtype)
        p_high = np.full((arr.shape[0], 1, 1), p_high, dtype=arr.dtype)

    denominator = np.maximum(p_high - p_low, 1e-6)
    arr = (arr - p_low) / denominator

    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)

    return arr.transpose(1, 2, 0)  # → [H, W, C]


def _safe_range(indices, max_idx):
    if not indices:
        return [max_idx // 2]  # 兜底取中间波段
    return [min(max(0, int(i)), max_idx - 1) for i in indices]


def _pick_bands(arr, bands):
    if isinstance(bands, list):
        return np.stack([arr[:, :, b] for b in bands], axis=-1)
    return arr[:, :, bands]


def _safe_bands(candidates, n_bands):
    valid = [b for b in candidates if b < n_bands]
    if not valid:
        valid = [n_bands - 1]
    while len(valid) < 3:
        valid.append(valid[-1])
    return valid[:3]


class Tester:
    def __init__(self, model, device, task, ckpt_path, physics, sigma, loss_type,
                 factor=None, standard="max", load_physics=True, ):
        self.model = model['model'].to(device)
        self.device = device
        self.task = task
        self.ckpt_path = ckpt_path
        self.physics = physics
        self.model_name = model['name']
        self.factor = factor
        self.metric = Metric(task=task, standard=standard, factor=factor)
        self.noisy = Denoising(GaussianNoise(sigma=sigma))
        self.loss_type = loss_type


        ckpt = torch.load(ckpt_path, map_location=device)

        self.model.res_block.load_state_dict(ckpt['model'])
        if load_physics:
            self.model.physics.load_state_dict(ckpt['physics'])

        print(f"Model Training Phase Best PSNR / NIQE: {ckpt['psnr']:.2f}")

    # ------------------------------------------------------------------
    # Inpainting
    # ------------------------------------------------------------------
    def test_inpainting(self, testloader, index, mat_index):
        x = testloader['data']
        name = testloader['name']
        n_bands = x.shape[1]

        candidates = [90, 60, 30, 10]
        bands = next((b for b in candidates if b < n_bands), n_bands // 2)
        save_path = f'./results/Inpainting/{name}/{self.model_name}/index{index}/mat{mat_index}/'
        os.makedirs(save_path, exist_ok=True)

        x = x.to(self.device)
        with torch.no_grad():
            y = self.physics(x)
            self.metric.compute(x, y)
            y_result = self.metric.average()
            print("Corrupted y  PSNR: {:.2f}, SSIM: {:.3f}, SAM: {:.3f}".format(*y_result))

            dagger = self.physics.A_adjoint(y)
            self.metric.compute(x, dagger)
            dagger_result = self.metric.average()
            print("Dagger       PSNR: {:.2f}, SSIM: {:.3f}, SAM: {:.3f}".format(*dagger_result))

            x1 = self.model(y)
            self.metric.compute(x, x1)
            result = self.metric.average()
            psnr, ssim, sam = result
            print("Recon        PSNR: {:.2f}, SSIM: {:.3f}, SAM: {:.3f}".format(*result))

        x_img = _pick_bands(_to_uint8(x), bands)
        y_img = _pick_bands(_to_uint8(y), bands)
        x1_img = _pick_bands(_to_uint8(x1), bands)
        dagger_img = _pick_bands(_to_uint8(dagger), bands)

        _save_img(os.path.join(save_path, 'gt.png'), x_img)
        _save_img(os.path.join(save_path, f'corrupted_{y_result[0]:.2f}_{y_result[1]:.3f}_{y_result[2]:.3f}.png'),
                  y_img)
        _save_img(os.path.join(save_path, f'recon_{psnr:.2f}_{ssim:.3f}_{sam:.3f}.png'), x1_img)
        _save_img(
            os.path.join(save_path, f'dagger_{dagger_result[0]:.2f}_{dagger_result[1]:.3f}_{dagger_result[2]:.3f}.png'),
            dagger_img)


    def test_sr(self, test_loader, sr_data_name="", patch_size=320, offset=(0, 0)):
        name = test_loader['name']
        sample = next(iter(test_loader['data']))
        n_bands = sample.shape[1]


        if name in ['Cave', 'CAVE']:
            bands_B, bands_G, bands_R = [5], [15], [25]

        elif name in ['PaviaUni', 'Pavia']:
            bands_B, bands_G, bands_R = [7], [29], [60]

        elif name in ['HeiPor', 'HeiPor2', 'HeiPor3']:
            bands_B = list(range(6, 13))  # 530-560 nm
            bands_G = list(range(8, 19))  # 540-590 nm
            bands_R = list(range(17, 46))  # 585-725 nm

        elif name in ['Agri', 'AgriFood', 'AgriFood2']:
            if 'wavelengths' in test_loader:
                wl = np.array(test_loader['wavelengths'])[:n_bands]
                bands_B = np.where((wl >= 530) & (wl <= 560))[0].tolist()
                bands_G = np.where((wl >= 540) & (wl <= 590))[0].tolist()
                bands_R = np.where((wl >= 585) & (wl <= 725))[0].tolist()
            else:
                bands_B = [int(n_bands * 0.2)]
                bands_G = [int(n_bands * 0.35)]
                bands_R = [int(n_bands * 0.65)]


        elif name in ['Brain1', 'Brain2', 'ThirdCampaign']:
            if 'wavelengths' in test_loader:
                wl = np.array(test_loader['wavelengths'])[:n_bands]

                bands_B = np.where((wl >= 470) & (wl <= 490))[0].tolist()
                bands_G = np.where((wl >= 530) & (wl <= 560))[0].tolist()
                bands_R = np.where((wl >= 680) & (wl <= 730))[0].tolist()
            else:
                bands_B = [int(n_bands * 0.13)]
                bands_G = [int(n_bands * 0.23)]
                bands_R = [int(n_bands * 0.50)]

        else:
            raise ValueError(f"Unknown dataset: {name}")

        # 确保索引合法
        bands_B = _safe_range(bands_B, n_bands)
        bands_G = _safe_range(bands_G, n_bands)
        bands_R = _safe_range(bands_R, n_bands)

        print(f"[Info] Visualization Bands -> B:{bands_B}, G:{bands_G}, R:{bands_R}")


        if name in ['Cave', 'CAVE']:
            save_path = f'./results/sr/{sr_data_name}/{self.model_name}/x{self.factor}'
        else:
            save_path = (f'./results/sr/{name}/patch{patch_size}_{offset[0]}_{offset[1]}'
                         f'/{self.model_name}/x{self.factor}')
        os.makedirs(save_path, exist_ok=True)
        print("save_path:", save_path)


        for x in test_loader['data']:
            x = x.to(self.device)
            with torch.no_grad():
                y = self.physics(x)

                dagger = F.interpolate(y, scale_factor=self.factor, mode='bicubic', align_corners=True)
                self.metric.compute(x, dagger)
                dagger_result = self.metric.average()
                print("Bicubic PSNR: {:.2f}, SSIM: {:.3f}, SAM: {:.2f}, ERGAS: {:.2f}".format(*dagger_result))

                x1 = self.model(y)
                self.metric.compute(x, x1)
                result = self.metric.average()
                psnr, ssim, sam, ergas = result
                print("Recon   PSNR: {:.2f}, SSIM: {:.3f}, SAM: {:.3f}, ERGAS: {:.2f}".format(psnr, ssim, sam, ergas))

            # --- spatial error map ---
            x_np = np.clip(x.squeeze().detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
            x1_np = np.clip(x1.squeeze().detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
            mean_spectral_error = np.mean(np.abs(x_np - x1_np), axis=2) * 255
            _save_img(os.path.join(save_path, 'share_spatial_error_map.png'), mean_spectral_error, cmap='jet', vmin=0,
                      vmax=40)


            def aggregate_to_rgb(tensor_data):
                R_mean = tensor_data[:, bands_R].mean(dim=1, keepdim=True)
                G_mean = tensor_data[:, bands_G].mean(dim=1, keepdim=True)
                B_mean = tensor_data[:, bands_B].mean(dim=1, keepdim=True)
                return torch.cat([R_mean, G_mean, B_mean], dim=1)

            x_rgb = aggregate_to_rgb(x)
            y_rgb = aggregate_to_rgb(y)
            x1_rgb = aggregate_to_rgb(x1)
            dagger_rgb = aggregate_to_rgb(dagger)

            x_img = _to_uint8(x_rgb, per_channel_stretch=True)
            y_img = _to_uint8(y_rgb, per_channel_stretch=True)
            x1_img = _to_uint8(x1_rgb, per_channel_stretch=True)
            dagger_img = _to_uint8(dagger_rgb, per_channel_stretch=True)

            _save_img(os.path.join(save_path, 'gt.png'), x_img)
            _save_img(os.path.join(save_path, 'lr.png'), y_img)
            _save_img(os.path.join(save_path, f'recon_{psnr:.2f}_{ssim:.3f}_{sam:.4f}_ergas{ergas:.2f}.png'), x1_img)
            _save_img(os.path.join(save_path,
                                   f'dagger_{dagger_result[0]:.2f}_{dagger_result[1]:.3f}_{dagger_result[2]:.3f}.png'),
                      dagger_img)


    def test_sr_real(self, test_loader, patch_size, offset):
        bands = [0, 1, 2]
        name = test_loader['name']
        save_path = (f'./results/sr_real/{name}/patch{patch_size}_{offset[0]}_{offset[1]}'
                     f'/{self.model_name}/x{self.factor}/{self.loss_type}')
        os.makedirs(save_path, exist_ok=True)
        print(f"Test Phase Real SR Noise Level: {self.physics.noise_model.sigma}")

        for lr_clean in test_loader['data']:
            lr_clean = lr_clean.to(self.device)
            y = self.physics.noise_model(lr_clean)

            with torch.no_grad():
                dagger = F.interpolate(y, scale_factor=self.factor, mode='bicubic')
                x1 = self.model(y)

            x1_img = _pick_bands(_to_uint8(x1), bands)
            y_img = _pick_bands(_to_uint8(y), bands)
            dagger_img = _pick_bands(_to_uint8(dagger), bands)
            lr_clean_img = _pick_bands(_to_uint8(lr_clean), bands)

            _save_img(os.path.join(save_path, 'lr.png'), y_img)
            _save_img(os.path.join(save_path, 'lr_clean.png'), lr_clean_img)
            _save_img(os.path.join(save_path, f'recon_x{self.factor}.png'), x1_img)
            _save_img(os.path.join(save_path,
                                   f'dagger_x{self.factor}.png'),
                      dagger_img)
            mat_path = os.path.join(save_path,
                                    f'recon_x{self.factor}_{self.loss_type}.mat')
            savemat(mat_path, {'HSI': x1[0].permute(1, 2, 0).cpu().numpy()})