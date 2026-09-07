import argparse
import random

import numpy as np
import torch
import torchvision.transforms as transforms

import deepinv as dinv
from data.physics import get_physics
# from models.spatial_attn import SHARE
from models.share import SHARE
from test import Tester
from train import Trainer
from utils_hsi import name_to_dict, transform_name_to_dict
from scipy.io import loadmat


LOSS_ZOO = [
    "surerec", "mcrec", "sureec", "mcec", "rec", "unsurerec", "r2rrec",
    "mc", "sure", "suretv", "unsure", "r2r",
]

DATASET_DEFAULTS = {
    "Cave":        {"in_channels": 31,  "mat_path": "./data/cave/",                                       "patch_size": 512},
    "PaviaUni":    {"in_channels": 103, "mat_path": "./data/Matzoo/PaviaU.mat",                           "patch_size": 320},
    "Chikusei_SR": {"in_channels": 128, "mat_path": "./data/Matzoo/HyperspecVNIR_Chikusei_20140729.mat",  "patch_size": 512},
    "TJS":         {"in_channels": 44,  "mat_path": "./data/TJS/HSI_44.mat",                              "patch_size": 256},
    "AgriFood":    {"in_channels": 75, "mat_path": "./data/AgriFood/UseCase_1_(Avoine1)_Anomaly_Easy_L9_6.bil.hdr", "patch_size": 480},
    "HeiPor":     {"in_channels": 100, "mat_path": "./data/HeiPorSPECTRAL/2021_04_15_09_22_02_SpecCube.dat", "patch_size": 480},
    "Brain1":      {"in_channels": 83, "mat_path": "./data/ThirdCampaignraw/042-01SecondCampaign.hdr", "patch_size": 480},
    "Brain2":      {"in_channels": 83, "mat_path": "./data/ThirdCampaignraw/043-01SecondCampaign.hdr", "patch_size": 480},
    "AgriFood2":    {"in_channels": 75, "mat_path": "./data/AgriFood/UseCase_1_(Avoine1)_Anomaly_Easy_L13_6.bil.hdr", "patch_size": 480},
}

FACTOR_DEFAULTS = {
    2: {"channel_dim": 64, "window_size": 8, "layers": 4},
    4: {"channel_dim": 64, "window_size": 8, "layers": 4},
    8: {"channel_dim": 64, "window_size": 8, "layers": 4},
    3: {"channel_dim": 64, "window_size": 8, "layers": 4},
}

MODEL_LOSS_OVERRIDE = {
    "hyperei":  "mcec",
    "R-Hy":     "mcec",
    "DeepHyIn": "mcec",
    "SSDL":     "suretv",
}


def parse_args():
    p = argparse.ArgumentParser(description="SHARE — Super Resolution")

    p.add_argument("--task", required=True,
                   choices=["sr", "test_sr", "sr_real", "test_sr_real",])
    p.add_argument("--dataset", required=True, choices=list(DATASET_DEFAULTS.keys()))
    p.add_argument("--model",   default="SHARE",
                   choices=["SHARE"])
    p.add_argument("--loss",    default="surerec", choices=LOSS_ZOO)
    p.add_argument("--factor",      type=int,   default=2,        choices=[2, 3, 4, 8])
    p.add_argument("--sigma",       type=float, default=25/255)
    p.add_argument("--sigma_real",  type=float, default=25/255)
    p.add_argument("--gain",        type=float, default=1/25)
    p.add_argument("--noise_type",  default="gaussian",
                   choices=["gaussian", "poisson", "gaussian_poisson"])

    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--epochs",      type=int,   default=800)
    p.add_argument("--alpha",       type=float, default=1.0)
    p.add_argument("--bs",          type=int,   default=1)

    # Architecture (0 = use factor-based default)
    p.add_argument("--channel_dim", type=int, default=0)
    p.add_argument("--window_size", type=int, default=0)
    p.add_argument("--layers",      type=int, default=0)

    p.add_argument("--transform",   default="Scale",
                   choices=["Rotate", "Shift", "Scale", "Reflect", "Affine", "Similarity", "Euclidean", "Tile",
                            "ScaleScale", "ShiftScaleScale", "SpectralScale", "BandPermutation"])
    p.add_argument("--n_trans",     type=int, default=3)

    p.add_argument("--sr_data_name", default=None,
                   help="Single scene .mat filename, or omit for 'all' in multi mode")
    p.add_argument("--patch_size",   type=int, default=0,   help="Override dataset default")
    p.add_argument("--offset",       type=int, nargs=2, default=[0, 0], metavar=("ROW", "COL"))
    p.add_argument("--rank", type=int, default=4, help="spectral rank for rank methods")
    p.add_argument("--memory_blocks", type=int, default=256, help="memory bank size")

    p.add_argument("--standard",    default="max")
    p.add_argument("--ckpt_step",   type=int, default=1500)
    p.add_argument("--ckpt",        default=None, help="Explicit checkpoint path")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--max_steps", type=int, default=50,
                   help="benchmark 时跑多少步（等价于 epoch 数）")
    p.add_argument('--run_time', type=int, default=1)
    p.add_argument("--retain_ratio", type=float, default=1.0,
                   help="Fraction of spectral bands to retain: 0.25, 0.5, 0.75, 1.0")
    return p.parse_args()



def build_ckpt_path(args, *, loss_type, transform_name, layers, channel_dim, patch_size):
    d, f, m = args.dataset, args.factor, args.model
    o  = f"{args.offset[0]}_{args.offset[1]}"
    lr, alpha = args.lr, args.alpha
    nt, nd = args.noise_type, args.sr_data_name
    sig, sigr, gain = args.sigma, args.sigma_real, args.gain
    rank = args.rank
    mem_size = args.memory_blocks
    ratio = args.retain_ratio

    if args.task in ("sr", "test_sr"):
        if d == "Cave":
            base = f"./checkpoints/ablation/sr/{nd}/{m}/x{f}/{loss_type}_{nt}/rank{rank}_mem{mem_size}/ratio{ratio:.2f}"
            if nt == "gaussian":
                fname = f"sr_BEST_data{d}_lr{lr}_alpha{alpha}_transform{transform_name}_sigma{sig}_layers{layers}_dim{channel_dim}.pth.tar"
            elif nt == "poisson":
                fname = f"sr_BEST_data{d}_lr{lr}_alpha{alpha}_transform{transform_name}_gain{gain}_layers{layers}_dim{channel_dim}.pth.tar"
            else:
                fname = f"sr_BEST_data{d}_lr{lr}_alpha{alpha}_transform{transform_name}_sigma{sig}_gain{gain}_layers{layers}_dim{channel_dim}.pth.tar"
        else:
            base = f"./checkpoints/ablation/sr/{d}/patch{patch_size}_{o}/{m}/x{f}/{loss_type}_{nt}/rank{rank}_mem{mem_size}/ratio{ratio:.2f}"
            fname = f"sr_BEST_data{d}_lr{lr}_alpha{alpha}_transform{transform_name}_sigma{sig}_layers{layers}_dim{channel_dim}.pth.tar"

        return f"{base}/{fname}"
    base  = f"./checkpoints/sr_real/{d}/patch{patch_size}_{o}/{m}/x{f}/{loss_type}_{nt}"
    fname = f"sr_real_epoch{10000}_psnr1.00_data{d}_lr{lr}_alpha{alpha}_transform{transform_name}_sigma{sigr}_layers{layers}_dim{channel_dim}.pth.tar"
    return f"{base}/{fname}"


def main():
    args = parse_args()
    seed_all(args.seed)

    ds          = DATASET_DEFAULTS[args.dataset]

    in_ch_full = ds["in_channels"]
    in_ch = int(round(in_ch_full * args.retain_ratio))
    print(f"Spectral bands: {in_ch_full} → {in_ch} (retain_ratio={args.retain_ratio})")

    mat_path    = ds["mat_path"]
    patch_size  = args.patch_size if args.patch_size > 0 else ds["patch_size"]
    sr_data_name = args.sr_data_name

    fd           = FACTOR_DEFAULTS[args.factor]
    channel_dim  = args.channel_dim  if args.channel_dim  > 0 else fd["channel_dim"]
    window_size  = args.window_size  if args.window_size  > 0 else fd["window_size"]
    layers       = args.layers       if args.layers       > 0 else fd["layers"]

    loss_type    = MODEL_LOSS_OVERRIDE.get(args.model, args.loss)
    is_real      = args.task in ("sr_real", "test_sr_real")
    sigma        = args.sigma_real if is_real else args.sigma

    device   = dinv.utils.get_freer_gpu()
    img_size = (44, patch_size * args.factor, patch_size * args.factor) if args.dataset == "TJS" else (in_ch, patch_size, patch_size)

    filter_params = loadmat('./Estimated_Responses.mat')['B']
    filter_tensor = torch.tensor(filter_params, dtype=torch.float32)
    if filter_tensor.dim() == 2:
        filter_tensor = filter_tensor.unsqueeze(0).unsqueeze(0)  # → (1, 1, h, w)
    filter_tensor = filter_tensor / filter_tensor.sum()
    # print(f"Loaded filter with shape {filter_tensor.shape} and sum {filter_tensor.sum().item():.4f}")

    physics  = get_physics(
        task=args.task, device=device, factor=args.factor,
        img_size=img_size, mat_index=1,          # mat_index unused for SR
        sigma=args.sigma, noise_type=args.noise_type, gain=args.gain,
        filter=filter_tensor if is_real else 'gaussian',
    )

    share = SHARE(
        in_channel=in_ch, window_size=window_size,
        physics=physics,
        layers=layers, channel_dim=channel_dim,
        rank=args.rank, memory_blocks=args.memory_blocks,
    )

    model_dict = {"model": share, "name": args.model}

    arg_loader = {
        "batch_size": args.bs, "task": args.task, "mode": "train",
        "single": True, "mat_path": mat_path,
        "sr_real_mat_path": DATASET_DEFAULTS["Cave"]["mat_path"],
        "device": device, "transform": transforms.Compose([transforms.ToTensor()]),
        "index": 0, "mat_index": 1,
        "sr_data_name": sr_data_name, "patch_size": patch_size,
        "offset": tuple(args.offset),
        "retain_ratio": args.retain_ratio,
    }
    train_loader, test_loader = name_to_dict(
        name=args.dataset, arg=arg_loader, task=args.task)

    transform_dict = transform_name_to_dict(args.transform, n_trans=args.n_trans)
    ckpt_path = args.ckpt or build_ckpt_path(
        args, loss_type=loss_type, transform_name=args.transform,
        layers=layers, channel_dim=channel_dim, patch_size=patch_size,
    )

    print(f"Task      : {args.task}  |  Dataset: {args.dataset}  |  Model: {args.model}")
    print(f"Loss      : {loss_type}  |  Transform: {args.transform}  |  Factor: x{args.factor}")
    print(f"Sigma     : {args.sigma:.4f}  |  Sigma_real: {args.sigma_real:.4f}  |  Noise: {args.noise_type}")
    print(f"Arch      : layers={layers}, dim={channel_dim}, window={window_size}")
    print(f"SR data   : {sr_data_name}")
    print(f"Ckpt      : {ckpt_path}")

    # ── Dispatch ──────────────────────────────
    if args.task in ("sr", "sr_real"):
        _train(args, model_dict, device, physics, train_loader, test_loader,
               transform_dict, loss_type, layers, channel_dim, sr_data_name, patch_size, sigma)
    elif args.task in ("test_sr", "test_sr_real"):
        _test(args, model_dict, device, physics, test_loader,
              ckpt_path, sr_data_name, patch_size, sigma, loss_type)


def _train(args, model_dict, device, physics, train_loader, test_loader,
           transform_dict, loss_type, layers, channel_dim, sr_data_name, patch_size, sigma):
    trainer = Trainer(
        task=args.task, device=device, epochs=args.epochs,
        lr=args.lr, alpha=args.alpha, factor=args.factor,
        ckpt_step=args.ckpt_step, standard=args.standard,
        sigma=sigma, index=0, mat_index=1, retain_ratio=args.retain_ratio,    # ← 加这行
    )
    trainer.setup(
        model=model_dict, trainloader=train_loader, testloader=test_loader,
        physics=physics, transform=transform_dict, loss_type=loss_type,
        layers=layers, channel_dim=channel_dim, resume=False, ckpt=None,
        sr_data_name=sr_data_name, patch_size=patch_size,
        offset=tuple(args.offset), noise_type=args.noise_type, gain=args.gain,
        rank=args.rank, memory_blocks=args.memory_blocks,
    )
    if args.task == "sr":
        trainer.train_sr()
    else:
        trainer.train_sr_real()


def _test(args, model_dict, device, physics, test_loader,
          ckpt_path, sr_data_name, patch_size, sigma, loss_type):
    tester = Tester(
        model=model_dict, device=device, task=args.task, physics=physics,
        ckpt_path=ckpt_path, sigma=sigma, factor=args.factor, standard=args.standard,
        loss_type=loss_type
    )
    if args.task == "test_sr":
        if args.dataset == "Cave":
            tester.test_sr(test_loader=test_loader, sr_data_name=sr_data_name)
        else:
            tester.test_sr(test_loader=test_loader, patch_size=patch_size)
    else:  # test_sr_real
        tester.test_sr_real(test_loader, patch_size=patch_size, offset=(0, 0))


# ──────────────────────────────────────────────
def seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    main()