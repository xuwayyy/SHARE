from deepinv.loss import SureGaussianLoss, EILoss, Loss, MCLoss
from deepinv.loss import SurePoissonLoss, SurePGLoss
from deepinv.physics import GaussianNoise, Denoising
from deepinv.loss import R2RLoss
import torch
from deepinv.loss import TVLoss




class SureRECLoss(Loss):
    def __init__(self, device, alpha, transform_ei, sigma=0.1, gain=1/20, noise_type="gaussian", unsure=False):
        """
        :param alpha: the weight coefficient, loss = sure + alpha * ei
        :param sigma: the gaussian noise level in physics, default=0.1
        :param transform_ei: transformation for ei loss
        """
        super(SureRECLoss, self).__init__()
        self.device = device
        if noise_type == "gaussian":
            self.sure = SureGaussianLoss(sigma=sigma, unsure=unsure)
        elif noise_type == "poisson":
            self.sure = SurePoissonLoss(gain=gain)
        elif noise_type == 'gaussian_poisson':
            self.sure = SurePGLoss(sigma=sigma, gain=gain, unsure=unsure)

        self.rec = EILoss(transform=transform_ei, weight=alpha)

    def forward(self, y, physics, model):
        x_net = model(y)
        noise_model = physics.noise_model
        loss_ei = self.rec(x_net=x_net, physics=physics, model=model).mean()
        physics.noise_model = GaussianNoise(0)
        loss_sure = self.sure(y=y, x_net=x_net, model=model, physics=physics).mean()
        physics.noise_model = noise_model
        loss = loss_ei + loss_sure
        return loss_sure, loss_ei, loss


class MCECLoss(Loss):
    def __init__(self, device, alpha, transform_ei):
        super(MCECLoss, self).__init__()
        self.device = device
        self.mc = MCLoss()
        self.ec = EILoss(transform=transform_ei, weight=alpha)

    def forward(self, y, physics, model):
        x_net = model(y)
        noise_model = physics.noise_model
        physics.noise_model = GaussianNoise(0)
        loss_mc = self.mc(y=y, x_net=x_net, physics=physics).mean()
        loss_ei = self.ei(x_net=x_net, physics=physics, model=model).mean()
        loss = loss_ei + loss_mc
        physics.noise_model = noise_model  # still re-load noise model, or in test observation has no noise
        return loss_mc, loss_ei, loss

class MCRECLoss(Loss):
    def __init__(self, device, alpha, transform_ei):
        """
        :param alpha: the weight coefficient, loss = sure + alpha * ei
        :param sigma: the gaussian noise level in physics, default=0.1
        :param transform_ei: transformation for ei loss
        """
        super(MCRECLoss, self).__init__()
        self.device = device
        self.mc = MCLoss()
        self.rec = EILoss(transform=transform_ei, weight=alpha)

    def forward(self, y, physics, model):
        x_net = model(y)
        noise_model = physics.noise_model
        loss_ei = self.ei(x_net=x_net, physics=physics, model=model).mean()
        physics.noise_model = GaussianNoise(0)
        loss_mc = self.mc(y=y, x_net=x_net, physics=physics).mean()
        physics.noise_model = noise_model
        loss = loss_ei + loss_mc
        return loss_mc, loss_ei, loss


class SureECLoss(Loss):
    def __init__(self, device, alpha, transform_ei, sigma=0.1, gain=1/20, noise_type="gaussian"):
        """
        :param alpha: the weight coefficient, loss = sure + alpha * ei
        :param sigma: the gaussian noise level in physics, default=0.1
        :param transform_ei: transformation for ei loss
        """
        super(SureECLoss, self).__init__()
        self.device = device
        if noise_type == "gaussian":
            self.sure = SureGaussianLoss(sigma=sigma)
        elif noise_type == "poisson":
            self.sure = SurePoissonLoss(gain=gain)
        elif noise_type == 'gaussian_poisson':
            self.sure = SurePGLoss(sigma=sigma, gain=gain)

        self.ec = EILoss(transform=transform_ei, weight=alpha)

    def forward(self, y, physics, model):
        x_net = model(y)
        noise_model = physics.noise_model
        physics.noise_model = GaussianNoise(0)
        loss_sure = self.sure(y=y, x_net=x_net, model=model, physics=physics).mean()
        loss_ei = self.ei(x_net=x_net, physics=physics, model=model).mean()
        physics.noise_model = noise_model
        loss = loss_ei + loss_sure
        return loss_sure, loss_ei, loss

class HandMCLoss(Loss):
    def __init__(self):
        super(HandMCLoss, self).__init__()
        self.mc = MCLoss()

    def forward(self, y, physics, model):
        x_net = model(y)
        noise_model = physics.noise_model
        physics.noise_model = GaussianNoise(0)
        loss_mc = self.mc(y=y, x_net=x_net, physics=physics).mean()
        physics.noise_model = noise_model
        return loss_mc


class SureTvLoss(Loss):
    def __init__(self, alpha=0.01, sigma=0.1):
        super(SureTvLoss, self).__init__()
        self.tv = TVLoss()
        self.sure = SureGaussianLoss(sigma=sigma)
        self.TV = TVLoss()
        self.alpha = alpha

    def forward(self, y, physics, model):
        x_net = model(y)
        noise_model = physics.noise_model
        physics.noise_model = GaussianNoise(0)
        loss_sure = self.sure(y=y, x_net=x_net, model=model, physics=physics).mean()
        loss_tv = self.TV(x_net).mean()
        physics.noise_model = noise_model
        loss = loss_sure + self.alpha * loss_tv
        return loss_sure, loss_tv, loss


class RECLoss(Loss):
    def __init__(self, device, alpha, transform_ei):
        super(RECLoss, self).__init__()
        self.ec = EILoss(transform=transform_ei, weight=alpha)
        self.device = device

    def forward(self, y, physics, model):
        x_net = model(y)
        loss = self.ec(x_net=x_net, physics=physics, model=model).mean()
        return loss


class ECLoss(Loss):
    def __init__(self, device, alpha, transform_ei):
        super(ECLoss, self).__init__()
        self.ec = EILoss(transform=transform_ei, weight=alpha)

    def forward(self, y, physics, model):
        x_net = model(y)
        noise_model = physics.noise_model
        physics.noise_model = GaussianNoise(0)
        loss = self.ec(x_net=x_net, physics=physics, model=model).mean()
        physics.noise_model = noise_model
        return loss


class SureLoss(Loss):
    def __init__(self, sigma=0.1, gain = 1/25,  unsure=False, noise_type="gaussian"):
        super(SureLoss, self).__init__()
        if noise_type == "gaussian":
            self.sure = SureGaussianLoss(sigma=sigma)
        elif noise_type == "poisson":
            self.sure = SurePoissonLoss(gain=gain)
        elif noise_type == 'gaussian_poisson':
            self.sure = SurePGLoss(sigma=sigma, gain=gain)

    def forward(self, y, physics, model):
        x_net = model(y)
        noise_model = physics.noise_model
        physics.noise_model = GaussianNoise(0)
        loss_sure = self.sure(y=y, x_net=x_net, model=model, physics=physics).mean()
        physics.noise_model = noise_model
        return loss_sure

class R2RRECLoss(Loss):
    def __init__(self, physics, device, alpha, transform_ei):
        super(R2RRECLoss, self).__init__()
        self.rec = RECLoss(alpha, transform_ei)
        self.r2r = R2RLoss(noise_model=physics.noise_model)

    def forward(self, y, physics, model):
        loss_rec = self.rec(y=y, physics=physics, model=model).mean()
        model = self.r2r.adapt_model(model)
        x_net = model(y, physics, update_parameters=True)
        loss_r2r = self.r2r(x_net, y=y, physics=physics, model=model).mean()
        loss = loss_rec + loss_r2r
        return  loss_r2r, loss_rec, loss


class UnsureLoss(Loss):
    def __init__(self, sigma=0.1, gain=1/25, noise_type='gaussian', unsure=True):
        super(UnsureLoss, self).__init__()
        if noise_type == "gaussian":
            self.unsure = SureGaussianLoss(sigma=sigma, unsure=True)
        elif noise_type == "poisson":
            self.unsure = SurePoissonLoss(gain=gain)
        elif noise_type == 'gaussian_poisson':
            self.unsure = SurePGLoss(sigma=sigma, gain=gain, unsure=True)

    def forward(self, y, physics, model):
        x_net = model(y)
        loss_unsure = self.unsure(y=y, x_net=x_net, model=model, physics=physics).mean()
        return loss_unsure

class HandR2RLoss(Loss):
    def __init__(self, physics):
        super(HandR2RLoss, self).__init__()
        self.r2r = R2RLoss(noise_model=physics.noise_model)
    
    def forward(self, y, physics, model):
        model = self.r2r.adapt_model(model)
        x_net = model(y, physics, update_parameters=True)
        loss_r2r = self.r2r(x_net, y=y, physics=physics, model=model).mean()
        return loss_r2r



