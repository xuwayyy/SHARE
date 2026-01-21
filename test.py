import torch
from metric import Metric
from matplotlib import pyplot as plt
import numpy as np
from deepinv.physics import Denoising, GaussianNoise
import torch.nn.functional as F
import os


class Tester:
    def __init__(self, model, device, task, ckpt_path, physics, sigma, factor=None, standard="max", load_physics=True):
        self.model = model['model'].to(device)
        self.device = device
        self.task = task
        self.ckpt_path = ckpt_path
        self.physics = physics
        self.model_name = model['name']
        self.factor = factor
        self.metric = Metric(task=task, standard=standard, factor=factor)
        self.noisy = Denoising(GaussianNoise(sigma=sigma))
        assert task in ['sr', 'inpainting', "test_sr", "test_inpainting"], 'task should be sr or inpainting'
        if task in ['sr', "test_sr"]:
            assert factor is not None, 'factor must be provided for SR task in Tester'
        ckpt = torch.load(ckpt_path, map_location=device)


        self.model.res_block.load_state_dict(
            ckpt['model'])
        if load_physics:
            self.model.physics.load_state_dict(ckpt['physics'])

        print(f"Model Training Phase Best PSNR / NIQE: {ckpt['psnr']:.2f}")

    def test_inpainting(self, testloader, index, mat_index):
        x = testloader['data']
        name = testloader['name']
        bands = 149 if name == 'Indian' else 90  # Indian: band good: 130, 139
        if name == 'Indian':  # Indian do not need index
            save_path = f'./results/Inpainting/{name}/{self.model_name}/mat{mat_index}/'
        else:
            save_path = f'./results/Inpainting/{name}/{self.model_name}/index{index}/mat{mat_index}/'
        os.makedirs(save_path, exist_ok=True)

        x = x.to(self.device)
        with torch.no_grad():
            y = self.physics(x)
            self.metric.compute(x, y)
            y_result = self.metric.average()
            print("Corrupted y PSNR: {:.2f}, SSIM: {:.3f}, SAM: {:.3f}".format(*y_result))

            dagger = self.physics.A_adjoint(y)
            self.metric.compute(x, dagger)
            dagger_result = self.metric.average()
            print("Dagger PSNR: {:.2f}, SSIM: {:.3f} SAM: {:.3f}".format(*dagger_result))

            x1 = self.model(y)
            self.metric.compute(x, x1)
            result = self.metric.average()
            psnr, ssim, sam = result
            print("Recon PSNR: {:.2f}, SSIM: {:.3f}, SAM: {:.3f}".format(*result))

            y_img = y.squeeze().detach().cpu().numpy()
            y_img = y_img.transpose((1, 2, 0))
            x_img = x.squeeze().detach().cpu().numpy()
            x_img = x_img.transpose((1, 2, 0))
            x1_img = x1.squeeze().detach().cpu().numpy()
            x1_img = x1_img.transpose((1, 2, 0))
            dagger_img = dagger.squeeze().detach().cpu().numpy()
            dagger_img = dagger_img.transpose((1, 2, 0))

            if isinstance(bands, list):
                x_img = np.stack([x_img[:, :, band] for band in bands], axis=-1)
                x1_img = np.stack([x1_img[:, :, band] for band in bands], axis=-1)
                y_img = np.stack([y_img[:, :, band] for band in bands], axis=-1)
                dagger_img = np.stack([dagger_img[:, :, band] for band in bands], axis=-1)
            else:
                x_img = x_img[:, :, bands]
                x1_img = x1_img[:, :, bands]
                y_img = y_img[:, :, bands]
                dagger_img = dagger_img[:, :, bands]

            # save gt
            plt.figure()
            plt.axis('off')
            plt.xticks()
            plt.yticks()
            plt.imshow(x_img, cmap='gray')
            plt.savefig(os.path.join(save_path, 'gt.png'), bbox_inches='tight', pad_inches=0, dpi=400)
            plt.close()

            # save y
            plt.figure()
            plt.axis('off')
            plt.xticks()
            plt.yticks()
            plt.imshow(y_img, cmap='gray')
            plt.savefig(os.path.join(save_path, f'corrupted_{y_result[0]:.2f}_{y_result[1]:.3f}_{y_result[2]:.3f}.png'),
                        bbox_inches='tight', pad_inches=0, dpi=400)
            plt.close()

            # save reconstruct
            plt.figure()
            plt.axis('off')
            plt.xticks()
            plt.yticks()
            plt.imshow(x1_img, cmap='gray')
            plt.savefig(os.path.join(save_path, f"recon_{psnr:.2f}_{ssim:.3f}_{sam:.3f}.png"),
                        bbox_inches='tight', pad_inches=0, dpi=400)
            plt.close()

            # save H^+y
            plt.figure()
            plt.axis('off')
            plt.xticks()
            plt.yticks()
            plt.imshow(dagger_img, cmap='gray')
            plt.savefig(os.path.join(save_path,
                                     f"dagger_{dagger_result[0]:.2f}_{dagger_result[1]:.3f}_{dagger_result[2]:.3f}.png"),
                        bbox_inches='tight', pad_inches=0, dpi=400)


    def test_sr(self, test_loader, sr_data_name="", patch_size=320, offset=(0, 0)):
        if test_loader['name'] == 'Cave':
            bands = [5, 15, 25]
        elif test_loader['name'] == 'PaviaUni':
            bands = [60, 29, 7]
        elif test_loader['name'] == 'Chikusei_SR':
            bands = [90]
        name = test_loader['name']

        if name == 'Cave':
            save_path = f'./results/sr/{sr_data_name}/{self.model_name}/x{self.factor}'
        else:
            save_path = f'./results/sr/{name}/patch{patch_size}_{offset[0]}_{offset[1]}/{self.model_name}/x{self.factor}'

        os.makedirs(save_path, exist_ok=True)
        print("save_path:", save_path)
        for x in test_loader['data']:
            x = x.to(self.device)
            with torch.no_grad():
                y = self.physics(x)
                dagger = F.interpolate(y, scale_factor=self.factor, mode='bicubic', align_corners=True)
                self.metric.compute(x, dagger)
                dagger_result = self.metric.average()
                print("Bicubic PSNR: {:.2f}, SSIM: {:.3f}, SAM: {:.2f}, EGRAS: {:.2f}".format(*dagger_result))

                x1 = self.model(y)
                self.metric.compute(x, x1)
                result = self.metric.average()
                psnr, ssim, sam, ergas = result

                x_np = x.squeeze().detach().cpu().numpy().transpose(1, 2, 0)  # GT [H,W,C]
                x1_np = x1.squeeze().detach().cpu().numpy().transpose(1, 2, 0)  # Recon [H,W,C]
                spatial_error = np.abs(x_np - x1_np)  # 重建误差
                bicubic_spatial_error = np.abs(x_np - dagger.squeeze().detach().cpu().numpy().transpose(1, 2, 0))  # bicubic error
                gt_self_error = x_np - x_np

                mean_spatial_error = np.mean(spatial_error, axis=2)
                mean_bicubic_error = np.mean(bicubic_spatial_error, axis=2)
                mean_spatial_error = (mean_spatial_error * 255).astype(np.uint8)
                mean_bicubic_error = (mean_bicubic_error * 255).astype(np.uint8)
                mean_gt_self_error = np.mean(gt_self_error, axis=2)

                plt.figure()
                plt.imshow(mean_spatial_error, cmap='jet', vmin=0, vmax=40)
                plt.xticks([]), plt.yticks([])
                plt.axis('off')

                plt.savefig(os.path.join(save_path, 'ssdl_spatial_error_map.png'),
                            bbox_inches='tight', dpi=400, pad_inches=0)
                plt.close()

                x_img = x.squeeze().detach().cpu().numpy() * 255
                x_img = x_img.transpose((1, 2, 0))
                x_img = x_img.astype(np.uint8)

                x1_img = x1.squeeze().detach().cpu().numpy() * 255
                x1_img = x1_img.transpose((1, 2, 0))
                x1_img = x1_img.astype(np.uint8)

                y_img = y.squeeze().detach().cpu().numpy() * 255
                y_img = y_img.transpose((1, 2, 0))
                y_img = y_img.astype(np.uint8)

                dagger_img = dagger.squeeze().detach().cpu().numpy() * 255
                dagger_img = dagger_img.transpose((1, 2, 0))
                dagger_img = dagger_img.astype(np.uint8)

                x_img = np.stack([x_img[:, :, band] for band in bands], axis=-1)
                x1_img = np.stack([x1_img[:, :, band] for band in bands], axis=-1)
                y_img = np.stack([y_img[:, :, band] for band in bands], axis=-1)
                dagger_img = np.stack([dagger_img[:, :, band] for band in bands], axis=-1)

                print("Recon PSNR: {:.2f}, SSIM: {:.3f}, SAM: {:.3f} EGRAS: {:.2f}".format(psnr, ssim, sam, ergas))

                # save gt
                plt.figure()
                plt.axis('off')
                plt.xticks()
                plt.yticks()
                plt.imshow(x_img, cmap='gray')
                plt.savefig(os.path.join(save_path, 'gt.png'), bbox_inches='tight', pad_inches=0, dpi=400)
                plt.close()

                # save y
                plt.figure()
                plt.axis('off')
                plt.xticks()
                plt.yticks()
                plt.imshow(y_img, cmap='gray')
                plt.savefig(
                    os.path.join(save_path, f'lr.png'),
                    bbox_inches='tight', pad_inches=0, dpi=400)
                plt.close()

                # save reconstruct
                plt.figure()
                plt.axis('off')
                plt.xticks()
                plt.yticks()
                plt.imshow(x1_img, cmap='gray')
                plt.savefig(os.path.join(save_path, f"recon_{psnr:.2f}_{ssim:.3f}_{sam:.4f}_ergas{ergas:.2f}.png"),
                            bbox_inches='tight', pad_inches=0, dpi=400)
                plt.close()

                # save H^+y
                plt.figure()
                plt.axis('off')
                plt.xticks()
                plt.yticks()
                plt.imshow(dagger_img, cmap='gray')
                plt.savefig(os.path.join(save_path,
                                         f"dagger_{dagger_result[0]:.2f}_{dagger_result[1]:.3f}_{dagger_result[2]:.3f}.png"),
                            bbox_inches='tight', pad_inches=0, dpi=400)
                plt.close()