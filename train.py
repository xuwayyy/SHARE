import torch
import torch.optim as optim
import os
from deepinv.loss import SureGaussianLoss, EILoss, Loss
from deepinv.physics import Denoising, GaussianNoise
from torch.utils.data import DataLoader
from metric import Metric
from tqdm import tqdm
from typing import OrderedDict
from deepinv.physics import Physics
from torch.optim import lr_scheduler
import torch.nn.functional as F
import math
import numpy as np
from loss import (SureRECLoss, MCRECLoss, SureECLoss, SureLoss, UnsureLoss, RECLoss,
                  MCECLoss, HandMCLoss, SureTvLoss, ECLoss, R2RRECLoss, HandR2RLoss)

                            

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        print('they are the same')
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def mpsnr(x_true, x_pred):
    n_bands = x_true.shape[1]
    batch_size = x_true.shape[0]
    mean_for_each = 0
    for i in range(batch_size):
        a = x_true[i, :, :, :]
        b = x_pred[i, :, :, :]
        p = [psnr(a[k, :, :], b[k, :, :]) for k in range(n_bands)]
        mean_for_each += np.mean(p)
    return mean_for_each / batch_size


def ssim(img1, img2, window_size=11, size_average=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, kernel_size=window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, kernel_size=window_size, stride=1, padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, kernel_size=window_size, stride=1, padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map


def mssim(x_true, x_pred, window_size=11, ):
    n_bands = x_true.shape[1]
    batch_size = x_true.shape[0]
    mean_for_each = 0
    for i in range(batch_size):
        a = x_true[i, :, :, :]
        b = x_pred[i, :, :, :]
        s = torch.stack(
            [ssim(a[k, :, :].unsqueeze(0).unsqueeze(0), b[k, :, :].unsqueeze(0).unsqueeze(0), window_size) for k in
             range(n_bands)])
        mean_for_each += s.mean().item()
    return mean_for_each / batch_size



class Trainer:
    def __init__(self, device, epochs, lr, alpha, task, standard, ckpt_step, factor=None, sigma=0.1, index=2, mat_index=1):
        self.model, self.transform, self.trainloader, self.testloader, self.scheduler, \
            self.criterion, self.optimizer, self.physics = [None] * 8
        # all these values initialized with None, assign in setup
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.factor = factor
        self.metric = Metric(task=task, factor=factor, standard=standard)
        self.sigma = sigma
        self.ckpt_step = ckpt_step
        self.task = task
        self.best_psnr = 9  # not save <= 20 state 
        self.best_niqe = 9999
        self.best_brisque = 9999
        self.index = index
        self.mat_index = mat_index
        self.noisy = Denoising(GaussianNoise(sigma=sigma))
        if task == 'sr' or task == "test_sr":
            assert factor is not None, "Run SR experiments But Factor Is None In Trainer"
        self.start_epoch = 0

    def setup(self, model: dict, transform: [dict, OrderedDict], trainloader: [dict, OrderedDict], testloader: [dict],
              physics: Physics, sr_data_name,  resume=False, ckpt=None, loss_type="sureei", layers=3, channel_dim=128,
              patch_size=256, offset=(0, 0), noise_type='gaussian', gain=1/20):
        """
        :param model: dict, keys: ['model', 'name']
        :param transform: dict, keys: ["transform', 'name'] for ei
        :param trainloader, testloader: dict, keys: ['data', 'name']
        """
        self.model = model['model']
        self.model_name = model['name']
        self.transform = transform
        self.trainloader = trainloader
        self.testloader = testloader
        self.loss_type = loss_type
        self.layers = layers
        self.channel_dim = channel_dim
        self.sr_data_name = sr_data_name
        self.patch_size = patch_size
        self.offset = list(offset)
        self.noise_type = noise_type
        self.gain = gain
        self.physics = physics
        self.resume = resume


        if loss_type == 'surerec':
            self.criterion = SureRECLoss(device=self.device, alpha=self.alpha, transform_ei=self.transform['transform'],
                                        sigma=self.sigma)
        elif loss_type == 'mcrec':
            self.criterion = MCRECLoss(device=self.device, alpha=self.alpha, transform_ei=self.transform['transform'],)
        elif loss_type == 'sureec':
            self.criterion = SureECLoss(device=self.device, alpha=self.alpha, transform_ei=self.transform['transform'],
                                        sigma=self.sigma)
        elif loss_type == 'mcec':
            self.criterion = MCECLoss(device=self.device, alpha=self.alpha, transform_ei=self.transform['transform'],)
        elif loss_type == 'rec':
            self.criterion = RECLoss(device=self.device, alpha=self.alpha, transform_ei=self.transform['transform'],)
        elif loss_type == 'ec':
            self.criterion = ECLoss(device=self.device, alpha=self.alpha, transform_ei=self.transform['transform'])
        elif loss_type == 'unsure':
            self.criterion = UnsureLoss(sigma=self.sigma, noise_type=self.noise_type, gain=self.gain)
        elif loss_type =='r2r':
            self.criterion = HandR2RLoss(physics=self.physics)
        elif loss_type == 'sure':
            self.criterion = SureLoss(sigma=self.sigma, noise_type=self.noise_type, gain=self.gain)
        
        elif loss_type == 'unsurerec':
            self.criterion = SureRECLoss(device=self.device, alpha=self.alpha, transform_ei=self.transform['transform'],
                                        sigma=self.sigma, unsure=True)
        elif loss_type == 'r2rrec':
            self.criterion = R2RRECLoss(device=self.device, physics=self.physics, transform_ei=self.transform['transform'], alpha=self.alpha)
            self.model = self.criterion.adapt_model(self.model)
        elif loss_type == "mc":
            self.criterion = HandMCLoss()
        elif loss_type == 'suretv':
            self.criterion = SureTvLoss(alpha=self.alpha, sigma=self.sigma)

        else:
            raise ValueError("loss_type must be 'sureei' or 'mcei' or 'mc', 'suretv")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-8)

        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,  # 余弦周期设为总训练轮数
            eta_min=self.optimizer.param_groups[0]['lr'] * 0.1  # 最终学习率是初始学习率的 0.1 倍
        )
        self.model.to(self.device)

        self.preckpt = ckpt
        if resume is True: 
            if self.model_name == "EHIR":
                assert ckpt is not None, 'resume training must provide previous ckpt'
                self.start_epoch = ckpt['epoch']

                if self.trainloader['name'] != 'PaviaUni':
                    self.best_psnr = ckpt['psnr']
                else:
                    self.best_niqe = ckpt['psnr']

                self.optimizer.load_state_dict(ckpt['optimizer'])
                self.model.res_block.load_state_dict(ckpt['model']) 
                self.model.physics.load_state_dict(ckpt['physics'])
                self.scheduler.load_state_dict(ckpt['scheduler'])
                print(f"⭐******** Resume Training from Epoch {self.start_epoch} ********")
            else:
                self.start_epoch = ckpt['epoch']
                self.best_psnr = ckpt['psnr']
                self.optimizer.load_state_dict(ckpt['optimizer'])
                self.scheduler.load_state_dict(ckpt['scheduler'])
                self.model.load_state_dict(ckpt['model'])
                print(f"⭐******** Resume Training from Epoch {self.start_epoch} ********")
    
            

    def test(self, epoch, save_path):
        # self.model.eval()
        psne_seq, ssim_seq = [], []
        with torch.no_grad():
            for x in self.testloader['data']:
                x = x.to(self.device)
                y = self.physics(x)
                x1 = self.model(y)
                # self.metric.compute(x, x1)
                psnr = mpsnr(x, x1)
                ssim = mssim(x, x1)
                psne_seq.append(psnr)
                ssim_seq.append(ssim)
                # self.metric.compute(x, x1)
            avg_psnr = np.mean(psne_seq)
            avg_ssim = np.mean(ssim_seq)

            # metric_result = self.metric.average()
            if self.task == 'sr':
                print(f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.3f}")
                # print("PSNR: {:.2f}, SSIM: {:.3f}, SAM: {:.2f}, ERGAS: {:.3f}".format(*metric_result))
            else:
                print(f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.3f}")
            psnr = avg_psnr
            if psnr > self.best_psnr:
                self.best_psnr = psnr
                self.save_model(epoch=epoch, psnr_niqe=psnr, save_best=True, save_path=save_path)

    def save_model(self, epoch, save_path, psnr_niqe, save_best=False):
        """
        save model at ckpt_interval step or best test set psnr occurred
        :param save_best: if True, save best test set psnr model and ignore epoch param
        """
        os.makedirs(save_path, exist_ok=True)
        if self.model_name in ['EHIR', 'PnP-DHP', 'DHP']:
            print("Saving EHIR Model State_dict")
            save_dict = {'model': self.model.res_block.state_dict(), 'psnr': float(f"{psnr_niqe:.2f}"),
                    'physics': self.model.physics.state_dict(), 'optimizer':self.optimizer.state_dict(), 
                    'scheduler':self.scheduler.state_dict(), 'epoch':epoch}
        else:
            save_dict = {'model':self.model.state_dict(), 'psnr':float(f"{psnr_niqe:.2f}"), 
            'optimizer':self.optimizer.state_dict(),'scheduler':self.scheduler.state_dict(), 'epoch':epoch}

        if not save_best:
            if self.task == 'inpainting':
                suffix = f"{self.task}_epoch{epoch}_data{self.trainloader['name']}_index{self.index}_mat{self.mat_index}_lr{self.lr}_alpha{self.alpha}_transform{self.transform['name']}_sigma{self.sigma}_layers{self.layers}_dim{self.channel_dim}.pth.tar"
            else:
                suffix = f"{self.task}_epoch{epoch}_psnr{psnr_niqe:.2f}_data{self.trainloader['name']}_lr{self.lr}_alpha{self.alpha}_transform{self.transform['name']}_sigma{self.sigma}_layers{self.layers}_dim{self.channel_dim}.pth.tar"
        else:
            if self.task == 'inpainting':
                if self.trainloader['name'] == 'Chikusei':
                    suffix = f"{self.task}_BEST_data{self.trainloader['name']}_index{self.index}_mat{self.mat_index}_lr{self.lr}_alpha{self.alpha}_transform{self.transform['name']}_sigma{self.sigma}_layers{self.layers}_dim{self.channel_dim}.pth.tar"
                else: # Indian 不需要保存index编号
                    suffix = f"{self.task}_BEST_data{self.trainloader['name']}_mat{self.mat_index}_lr{self.lr}_alpha{self.alpha}_transform{self.transform['name']}_sigma{self.sigma}_layers{self.layers}_dim{self.channel_dim}.pth.tar"

            else:
                if self.noise_type == 'gaussian':
                    suffix = f"{self.task}_BEST_data{self.trainloader['name']}_lr{self.lr}_alpha{self.alpha}_transform{self.transform['name']}_sigma{self.sigma}_layers{self.layers}_dim{self.channel_dim}.pth.tar"
                elif self.noise_type == 'poisson':
                    suffix = f"{self.task}_BEST_data{self.trainloader['name']}_lr{self.lr}_alpha{self.alpha}_transform{self.transform['name']}_gain{self.gain}_layers{self.layers}_dim{self.channel_dim}.pth.tar"
                elif self.noise_type == 'gaussian_poisson':
                    suffix = f"{self.task}_BEST_data{self.trainloader['name']}_lr{self.lr}_alpha{self.alpha}_transform{self.transform['name']}_sigma{self.sigma}_gain{self.gain}_layers{self.layers}_dim{self.channel_dim}.pth.tar"


        torch.save(save_dict, os.path.join(save_path, suffix))
        if save_best:
            print("✅ New Best Test Set Psnr / NIQE: {:.2f}".format(psnr_niqe))
        else:
            print(f"Saving model to {save_path}/{suffix}")

    def train_sr(self):
        self.model.train()
        assert self.task == 'sr', 'run train_sr but task is inpainting'

        if self.trainloader['name'] == 'Cave':
            save_path = f"./checkpoints/sr/{self.sr_data_name}/{self.model_name}/x{self.factor}/{self.loss_type}_{self.noise_type}"
        else:
            save_path = f"./checkpoints/sr/{self.trainloader['name']}/patch{self.patch_size}_{self.offset[0]}_{self.offset[1]}/{self.model_name}/x{self.factor}/{self.loss_type}_{self.noise_type}"
        
        assert isinstance(self.trainloader['data'], DataLoader), "current trainloader is not a DataLoader, maybe you forget modify task and data name"
        psnr_seq, ssim_seq = [], []

        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            for i, x in enumerate(tqdm(self.trainloader['data'])):
                x = x.to(self.device)
                y = self.physics(x)

                x_net = self.model(y)
                if self.loss_type not in ["mc", "sure", "ei", "unsure", "r2r"]:
                    loss_sure, loss_ei, loss = self.criterion(y=y, physics=self.physics, model=self.model)
                    print(
                    f'Epoch: {epoch + 1}, Sure Loss: {loss_sure.item():.3f}, EI Loss: {loss_ei.item():.3f}, Loss: {loss.item():.3f}')
                else:
                    loss = self.criterion(y=y, physics=self.physics, model=self.model)
                    print( f"Epoch: {epoch + 1}, loss: {loss.item():.3f}")
                psnr = mpsnr(x_net, x)
                ssim = mssim(x_net, x)
                psnr_seq.append(psnr)
                ssim_seq.append(ssim)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # print(
            #     f'Epoch: {epoch + 1}, Sure Loss: {loss_sure.item():.3f}, EI Loss: {loss_ei.item():.3f}, Loss: {loss.item():.3f}', end='')
            avg_psnr, avg_ssim = np.mean(psnr_seq), np.mean(ssim_seq)
            if (epoch + 1) % self.ckpt_step == 0 or epoch == self.epochs - 1:
                self.save_model(epoch=epoch + 1, psnr_niqe=avg_psnr, save_path=save_path, save_best=False)
            if self.model_name != 'SSDL':
                self.scheduler.step()
            self.test(epoch=epoch + 1, save_path=save_path)


    def train_inpainting(self):
        
        assert self.task == 'inpainting', 'run train_inpainting but task is sr'
        save_path = f"./checkpoints/inpainting/{self.model_name}/{self.loss_type}/"

        assert isinstance(self.trainloader['data'], torch.Tensor), "current trainloader is a DataLoader, maybe you pass Inpainting Dataset"
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            x = self.trainloader['data'].to(self.device)
            y = self.physics(x)
            x_net = self.model(y)
            if self.loss_type != "mc":
                loss_sure, loss_ei, loss = self.criterion(y=y, physics=self.physics, model=self.model)
            else:
                loss = self.criterion(y=y, physics=self.physics, model=self.model)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.loss_type != "mc":
                print(
                    f'Epoch: {epoch + 1}, Sure Loss: {loss_sure.item():.3f}, EI Loss: {loss_ei.item():.3f}, Loss: {loss.item():.3f}')
            else:
                print(
                    f"Epoch: {epoch + 1}, loss: {loss.item():.3f}"
                )
            psnr = mpsnr(x_net, x)
            ssim = mssim(x_net, x)
            print(f" PSNR: {psnr:.2f}, SSIM: {ssim:.3f}")
            if (epoch + 1) % self.ckpt_step == 0 or epoch == self.epochs - 1:
                self.save_model(epoch=epoch + 1, psnr=psnr, save_path=save_path, save_best=False)
            if psnr > self.best_psnr:
                self.best_psnr = psnr
                self.save_model(epoch=epoch + 1, psnr_niqe=psnr, save_path=save_path, save_best=True)
            self.scheduler.step()
