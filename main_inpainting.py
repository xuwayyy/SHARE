import argparse
import random

import numpy as np
import torch
import torchvision.transforms as transforms

import deepinv as dinv
from data.physics import get_physics
from models.share import SHARE
from test import Tester
from train import Trainer
from utils_hsi import name_to_dict, transform_name_to_dict



LOSS_ZOO = [
    "surerec", "mcrec", "sureec", "mcec", "rec", "unsurerec", "r2rrec",
    "mc", "sure", "suretv", "unsure", "r2r",
]

DATASET_DEFAULTS = {
    "Chikusei": {"in_channels": 128, "mat_path": ""},
    "Indian":   {"in_channels": 200, "mat_path": ""},
}

MODEL_LOSS_OVERRIDE = {
    "hyperei":  "mcec",
    "R-Hy":     "mcec",
    "DeepHyIn": "mcec",
}

WINDOW_SIZE  = 6
LAYERS       = 4
CHANNEL_DIM  = 128



def parse_args():
    p = argparse.ArgumentParser(description="SHARE — Inpainting")

    p.add_argument("--task",    required=True, choices=["inpainting", "test_inpainting"])
    p.add_argument("--dataset", required=True, choices=list(DATASET_DEFAULTS.keys()))
    p.add_argument("--model",   default="SHARE",
                   choices=["SHARE"])
    p.add_argument("--loss",    default="sureei", choices=LOSS_ZOO)

    p.add_argument("--sigma",      type=float, default=25/255)
    p.add_argument("--gain",       type=float, default=1/25)
    p.add_argument("--noise_type", default="gaussian",
                   choices=["gaussian", "poisson", "gaussian_poisson"])
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--epochs",     type=int,   default=800)
    p.add_argument("--alpha",      type=float, default=1.0)
    p.add_argument("--bs",         type=int,   default=1)

    p.add_argument("--mat_index",  type=int, default=3, choices=[1, 2, 3, 4],
                   help="Inpainting mask type")
    p.add_argument("--index",      type=int, default=4,
                   help="Chikusei scene index [0-4]")
    p.add_argument("--rank", type=int, default=4, help="spectral rank for rank methods")
    p.add_argument("--memory_blocks", type=int, default=256, help="memory bank size")

    p.add_argument("--transform",  default="Shift",
                   choices=["Rotate", "Shift", "Scale", "Reflect", "Affine", "Similarity", "Euclidean", "Tile",
                            "InpaintingShiftScale", "InpaintingRotateShift","InpaintingReflectScale", "BandPermutation" ])
    p.add_argument("--n_trans",    type=int, default=3)

    p.add_argument("--standard",      default="normal")
    p.add_argument("--ckpt_step",     type=int, default=50000)
    p.add_argument("--ckpt",          default=None, help="Explicit checkpoint path")
    p.add_argument("--load_physics",  action="store_true", default=True)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--max_steps", type=int, default=50,)
    p.add_argument('--run_time', type=int, default=1)
    p.add_argument("--retain_ratio", type=float, default=1.0,
                   help="Fraction of spectral bands to retain: 0.25, 0.5, 0.75, 1.0")

    return p.parse_args()



def build_ckpt_path(args, *, loss_type):
    m, d = args.model, args.dataset
    lr, alpha, sigma = args.lr, args.alpha, args.sigma
    idx, mat = args.index, args.mat_index
    t  = args.transform
    nt = args.noise_type
    gain = args.gain
    rank = args.rank
    mem_size = args.memory_blocks
    ratio = args.retain_ratio

    print(f"Rank: {rank}, Memory Blocks: {mem_size}, Ratio: {ratio}")


    base = f"./checkpoints/ablation/inpainting/{m}/{loss_type}_{nt}/rank{rank}_mem{mem_size}"

    if d == "Chikusei":
        if nt == "gaussian":
            fname = (f"inpainting_BEST_dataChikusei_index{idx}_mat{mat}"
                     f"_lr{lr}_alpha{alpha}_transform{t}"
                     f"_sigma{sigma}"
                     f"_layers{LAYERS}_dim{CHANNEL_DIM}.pth.tar")
        elif nt == "poisson":
            fname = (f"inpainting_BEST_dataChikusei_index{idx}_mat{mat}"
                     f"_lr{lr}_alpha{alpha}_transform{t}"
                     f"_gain{gain}"
                     f"_layers{LAYERS}_dim{CHANNEL_DIM}.pth.tar")
        else:
            fname = (f"inpainting_BEST_dataChikusei_index{idx}_mat{mat}"
                     f"_lr{lr}_alpha{alpha}_transform{t}"
                     f"_sigma{sigma}_gain{gain}"
                     f"_layers{LAYERS}_dim{CHANNEL_DIM}.pth.tar")

    return f"{base}/{fname}"



def main():
    args = parse_args()
    seed_all(args.seed)


    ds       = DATASET_DEFAULTS[args.dataset]
    in_ch_full = ds["in_channels"]  # e.g. 128 for Chikusei
    mat_path = ds["mat_path"]
    loss_type = MODEL_LOSS_OVERRIDE.get(args.model, args.loss)

    in_ch = int(round(in_ch_full * args.retain_ratio))
    print(f"Spectral bands: {in_ch_full} → {in_ch} (retain_ratio={args.retain_ratio})")


    device   = dinv.utils.get_freer_gpu()
    img_size = (in_ch, 512, 512)
    physics  = get_physics(
        task=args.task, device=device, factor=1,
        img_size=img_size, mat_index=args.mat_index,
        sigma=args.sigma, noise_type=args.noise_type, gain=args.gain,
    )


    share = SHARE(
        in_channel=in_ch, window_size=WINDOW_SIZE,
        physics=physics, layers=LAYERS, channel_dim=CHANNEL_DIM,
        rank=args.rank, memory_blocks=args.memory_blocks,
    )

    model_dict = {"model": share, "name": args.model}


    arg_loader = {
        "batch_size": args.bs, "task": args.task, "mode": "train",
        "single": False, "mat_path": mat_path, "sr_real_mat_path": "",
        "device": device, "transform": transforms.Compose([transforms.ToTensor()]),
        "index": args.index, "mat_index": args.mat_index,
        "sr_data_name": "", "patch_size": 512,
        "offset": (0, 0), "sr_mode": "multi",
        "retain_ratio": args.retain_ratio,
    }
    train_loader, test_loader = name_to_dict(
        name=args.dataset, arg=arg_loader, task=args.task)


    transform_dict = transform_name_to_dict(args.transform, n_trans=args.n_trans)
    ckpt_path = args.ckpt or build_ckpt_path(args, loss_type=loss_type)


    print(f"Task      : {args.task}  |  Dataset: {args.dataset}  |  Model: {args.model}")
    print(f"Loss      : {loss_type}  |  Transform: {args.transform}")
    print(f"Sigma     : {args.sigma:.4f}  |  Gain: {args.gain:.4f}  |  Noise: {args.noise_type}")
    print(f"Mask      : mat_index={args.mat_index}")
    print(f"Arch      : layers={LAYERS}, dim={CHANNEL_DIM}, window={WINDOW_SIZE}")
    print(f"Ckpt      : {ckpt_path}")


    if args.task == "inpainting":
        _train(args, model_dict, device, physics, train_loader, test_loader,
               transform_dict, loss_type)
    elif args.task == 'test_inpainting':
        _test(args, model_dict, device, physics, test_loader, ckpt_path, loss_type)


def _train(args, model_dict, device, physics, train_loader, test_loader,
           transform_dict, loss_type):
    trainer = Trainer(
        task=args.task, device=device, epochs=args.epochs,
        lr=args.lr, alpha=args.alpha, factor=1,
        ckpt_step=args.ckpt_step, standard=args.standard,
        sigma=args.sigma, index=args.index, mat_index=args.mat_index,
        retain_ratio=args.retain_ratio,
    )
    trainer.setup(
        model=model_dict, trainloader=train_loader, testloader=test_loader,
        physics=physics, transform=transform_dict, loss_type=loss_type,
        layers=LAYERS, channel_dim=CHANNEL_DIM, resume=False, ckpt=None,
        sr_data_name="", patch_size=512, offset=(0, 0),
        noise_type=args.noise_type, gain=args.gain, rank=args.rank, memory_blocks=args.memory_blocks,
    )
    trainer.train_inpainting()


def _test(args, model_dict, device, physics, test_loader, ckpt_path, loss_type):
    tester = Tester(
        model=model_dict, device=device, task=args.task, physics=physics,
        ckpt_path=ckpt_path, sigma=args.sigma, factor=1,
        standard=args.standard, load_physics=args.load_physics,
        loss_type=loss_type

    )
    tester.test_inpainting(test_loader, args.index, args.mat_index)



def seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    main()