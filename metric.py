import torch
import numpy as np
from deepinv.loss.metric import PSNR, SSIM, SpectralAngleMapper, ERGAS,NIQE


def mpsnr(x_true, x_pred, standard="normal"):
    psnr = PSNR()
    x_pred_norm = (x_pred - x_pred.min()) / (x_pred.max() - x_pred.min() + 1e-6)
    n_bands = x_true.shape[1]
    batch_size = x_true.shape[0]
    mean_for_each = 0
    for i in range(batch_size):
        a = x_true[i, :, :, :]
        b = x_pred[i, :, :, :]
        c = x_pred_norm[i, :, :, :]
        p = [psnr(a[k, :, :], b[k, :, :]) for k in range(n_bands)]
        p_norm = [psnr(a[k, :, :], c[k, :, :]) for k in range(n_bands)]
        if standard == "max":
            mean_for_each += max(np.mean(p_norm), np.mean(p))
        elif standard == "min":
            mean_for_each += min(np.mean(p_norm), np.mean(p))
        elif standard == "normal":
            mean_for_each += np.mean(p)
        else:
            raise ValueError("Standard must be either 'max' or 'min'")
    return mean_for_each / batch_size


def mssim(x_true, x_pred, standard="max"):
    ssim = SSIM()
    n_bands = x_true.shape[1]
    batch_size = x_true.shape[0]
    x_pred_norm = (x_pred - x_pred.min()) / (x_pred.max() - x_pred.min() + 1e-6)
    mean_for_each = 0
    for i in range(batch_size):
        a = x_true[i, :, :, :]
        b = x_pred[i, :, :, :]
        c = x_pred_norm[i, :, :, :]
        p = [ssim(a[k, :, :].unsqueeze(0).unsqueeze(0), b[k, :, :].unsqueeze(0).unsqueeze(0)) for k in range(n_bands)]
        p_norm = [ssim(a[k, :, :].unsqueeze(0).unsqueeze(0), c[k, :, :].unsqueeze(0).unsqueeze(0)) for k in
                  range(n_bands)]
        if standard == "max":
            mean_for_each += max(np.mean(p_norm), np.mean(p))
        elif standard == "min":
            mean_for_each += min(np.mean(p), np.mean(p_norm))
        elif standard == "normal":
            mean_for_each += np.mean(p)
        else:
            raise ValueError("Standard must be either 'max' or 'min'")
    return mean_for_each / batch_size



def SpectralAngle(x_true, x_pred, standard="max"):
    cal = SpectralAngleMapper()
    sam = cal(x_true, x_pred)
    x_pred_norm = (x_pred - x_pred.min()) / (x_pred.max() - x_pred.min() + 1e-6)
    sam_norm = cal(x_true, x_pred_norm)
    if standard == "max":
        return max(sam.mean(), sam_norm.mean())
    elif standard == "min":
        return min(sam.mean(), sam_norm.mean())
    elif standard == "normal":
        return sam.mean()
    else:
        raise ValueError("Standard must be either 'max' or 'min'")



class Metric:
    def __init__(self, task, standard="normal", factor=None):
        super().__init__()
        self.psnr_seq = []
        self.ssim_seq = []
        self.sam_seq = []
        self.standard = standard
        self.task = task
        self.factor = factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cal_niqe = NIQE(device=self.device, check_input_range=True, norm_inputs="min_max")
        if task == "sr" or task == "test_sr":
            assert factor is not None, "Factor must be provided for SR task in init for ERGAS metric"
        if (task == "inpainting" or task == 'test_inpainting') and factor is not None:
            Warning("Remember to update factor when run SR experiments")


    def compute(self, x_true, x_pred):

        if x_true.is_cuda and x_pred.is_cuda:
            x_true = x_true.detach().cpu()  # still need tensor not numpy
            x_pred = x_pred.detach().cpu()

        result_psnr = mpsnr(x_true, x_pred, standard=self.standard)
        result_ssim = mssim(x_true, x_pred, standard=self.standard)
        result_sam = SpectralAngle(x_true, x_pred, standard=self.standard)
        self.psnr_seq.append(result_psnr)
        self.ssim_seq.append(result_ssim)
        self.sam_seq.append(result_sam)


    def average(self):

        self.sam_seq = [x for x in self.sam_seq if not np.isnan(x)]
        psnr, ssim, sam = np.mean(self.psnr_seq), np.mean(self.ssim_seq), np.mean(self.sam_seq)
        # sam = (sam * 180) / np.pi  # radian to degree
        self.psnr_seq, self.ssim_seq, self.sam_seq = [], [], []  # clear this epoch value and prepare for next epoch
        if self.task == "sr" or self.task == "test_sr":
            ergas = np.mean(self.ergas_seq)
            self.ergas_seq = []
            return psnr, ssim, sam, ergas
        else:
            return psnr, ssim, sam



if __name__ == '__main__':
    device = "cuda:0"
    x1 = torch.ones(2, 31, 256, 256).to(device)
    x2 = torch.ones(2, 31, 256, 256).to(device)
    task = "inpainting"
    metric = Metric(task="inpainting", factor=2)
    metric.compute(x1, x2)
    if task == "sr":
        print("Epoch: {}, PSNR: {:.2f}, SSIM: {:.3f}, SAM: {:.2f}, ERGAS: {:.3f}".format(1, *metric.average()))
    else:
        print("Epoch: {}, PSNR: {:.2f}, SSIM: {:.3f}, SAM: {:.2f}".format(1, *metric.average()))