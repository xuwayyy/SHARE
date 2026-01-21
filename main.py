from test import Tester
from train import Trainer
from models.share import Share, ResBlock
from data.physics import get_physics
from utils_hsi import name_to_dict, transform_name_to_dict, get_baseline

import deepinv as dinv

import torch
import numpy as np
import random
import argparse
from scipy.io import loadmat
import torchvision.transforms as transforms


def build_parser():
    parser = argparse.ArgumentParser(
        description="Share for HSI SR / Inpainting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ===================== basic =====================
    parser.add_argument('--task', type=str, default='test_sr',
                        choices=['sr', 'inpainting', 'test_sr', 'test_inpainting'])

    parser.add_argument('--dataset', type=str, default='Cave',
                        choices=['Cave', 'Indian', 'Chikusei', 'PaviaUni'])

    parser.add_argument('--model', type=str, default='Share')

    parser.add_argument('--loss', type=str, default="surerec",
                        choices=["surerec", "mcrec", "sureec", "mcec", "mc", "sure", "rec", "sureei"])

    parser.add_argument('--seed', type=int, default=42)

    # ===================== SR / Inpainting =====================
    parser.add_argument('--factor', type=int, default=2, help="SR downsample factor")
    parser.add_argument('--patch_size', type=int, default=512, help="crop size for SR task for Chikusei / PaviaUni")
    parser.add_argument('--sr_data_name', type=str, default='fake_and_real_beers_ms.mat',
                        help="When specific Cave, use which HSI mat",
                        choices=['glass_tiles_ms.mat', 'fake_and_real_beers_ms.mat'])
    parser.add_argument('--index', type=int, default=4)
    parser.add_argument('--mat_index', type=int, default=3)

    # ===================== training =====================
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--lr', type=float, default=1e-2, help="1e-2 for inpainting, 1e-3 for SR")
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--ckpt_step', type=int, default=1000)

    # ===================== noise =====================
    parser.add_argument('--noise_type', type=str, default='gaussian',
                        choices=['gaussian', 'poisson', 'gaussian_poisson'])
    parser.add_argument('--sigma', type=float, default=25/255, help="Gaussian noise sigma")
    parser.add_argument('--gain', type=float, default=1/25, help="Poisson Gain factor")

    # ===================== equivariant =====================
    parser.add_argument('--transform', type=str, default='Scale',
                        choices=['Rotate', 'Shift', 'Scale', 'Reflect',
                                 'Affine', 'Similarity', 'Euclidean', 'Tile'])
    parser.add_argument('--n_trans', type=int, default=3)

    # ===================== misc =====================
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default=None)


    return parser


def postprocess_args(args):
    # ---------- dataset ----------
    mat_path_map = {
        'Cave': './data/cave/',
        'Indian': '',
        'Chikusei': '',
        'PaviaUni': './data/Matzoo/PaviaU.mat',
        'Chikusei_SR': './data/Matzoo/HyperspecVNIR_Chikusei_20140729.mat'
    }

    in_channels_map = {
        'Cave': 31,
        'Indian': 200,
        'Chikusei': 128,
        'PaviaUni': 103,
        'Chikusei_SR': 128
    }

    args.in_channels = in_channels_map[args.dataset]
    args.mat_path = mat_path_map[args.dataset]

    if args.dataset == 'PaviaUni':
        args.patch_size = 320

    # ---------- task related ----------
    if args.task in ['sr', 'test_sr']:
        args.channel_dim = 32 if args.factor in [2, 4] else 16
        args.layers = 3 if args.factor <= 4 else 2
        args.window_size = {2: 8, 4: 4, 8: 2}[args.factor]
    else:
        args.channel_dim = 128
        args.layers = 4
        args.window_size = 6

    return args


def setup_model(args, physics):
    """Initialize the model based on args"""
    model = Share(
        in_channel=args.in_channels,
        window_size=args.window_size,
        physics=physics,
        layers=args.layers,
        channel_dim=args.channel_dim,
    )
    model_dict = {'model': model, 'name': args.model}
    return model_dict


def get_checkpoint_path(args):
    """Generate checkpoint path based on task and configuration"""

    base_params = f"data{args.dataset}_lr{args.lr}_alpha{args.alpha}_transform{args.transform}_sigma{args.sigma}_layers{args.layers}_dim{args.channel_dim}"

    if args.task == "test_sr":
        if args.dataset == 'Cave':
            if args.noise_type == 'gaussian':
                return f"./checkpoints/sr/{args.sr_data_name}/{args.model}/x{args.factor}/{args.loss}_{args.noise_type}/sr_BEST_{base_params}.pth.tar"
            elif args.noise_type == 'poisson':
                base_params = base_params.replace(f"sigma{args.sigma}", f"gain{args.gain}")
                return f"./checkpoints/sr/{args.sr_data_name}/{args.model}/x{args.factor}/{args.loss}_{args.noise_type}/sr_BEST_{base_params}.pth.tar"
            else:  # gaussian_poisson
                base_params = base_params.replace(f"sigma{args.sigma}", f"sigma{args.sigma}_gain{args.gain}")
                return f"./checkpoints/sr/{args.sr_data_name}/{args.model}/x{args.factor}/{args.loss}_{args.noise_type}/sr_BEST_{base_params}.pth.tar"
        else:
            return f"./checkpoints/sr/{args.dataset}/patch{args.patch_size}_{args.offset[0]}_{args.offset[1]}/{args.model}/x{args.factor}/{args.loss}/sr_BEST_{base_params}.pth.tar"

    elif args.task == "test_inpainting":
        base_params = f"data{args.dataset}_index{args.index}_mat{args.mat_index}_lr{args.lr}_alpha{args.alpha}_transform{args.transform}_sigma{args.sigma}_layers{args.layers}_dim{args.channel_dim}"
        if args.dataset == "Chikusei":
            return f"./checkpoints/inpainting/{args.model}/{args.loss}/inpainting_BEST_{base_params}.pth.tar"
        else:
            base_params = base_params.replace(f"index{args.index}_", "")
            return f"./checkpoints/inpainting/{args.model}/{args.loss}/inpainting_BEST_{base_params}.pth.tar"


    return args.ckpt_path


def main():
    parser = build_parser()
    args = parser.parse_args()

    args = postprocess_args(args)

    seed_all(args.seed)

    device = dinv.utils.get_freer_gpu()
    transform = transforms.Compose([transforms.ToTensor()])

    data_config = {
        'batch_size': args.batch_size,
        'task': args.task,
        'mode': 'train', # do not specify this
        'single': True,
        'mat_path': args.mat_path,
        'device': device,
        'transform': transform,
        'index': args.index,
        'mat_index': args.mat_index,
        'sr_data_name': args.sr_data_name,
        'patch_size': args.patch_size,
    }

    train_config = {
        'lr': args.lr,
        'alpha': args.alpha,
        'epochs': args.epochs,
        'ckpt_step': args.ckpt_step,
        'factor': args.factor,
        'sigma': args.sigma,
        'standard': 'normal',
    }

    model_config = {
        'in_channels': args.in_channels,
        'out_channels': args.in_channels,
        'window_size': args.window_size,
        'filter': True,
        'img_size': (args.in_channels, args.patch_size, args.patch_size),
        'layers': args.layers,
        'channel_dim': args.channel_dim,
    }

    # Merge all configurations
    config = {**data_config, **train_config, **model_config}

    physics = get_physics(
        task=args.task,
        device=device,
        factor=args.factor,
        img_size=config['img_size'],
        mat_index=args.mat_index,
        sigma=args.sigma,
        noise_type=args.noise_type,
        gain=args.gain
    )

    model_dict = setup_model(args, physics)

    transform_dict = transform_name_to_dict(args.transform, n_trans=args.n_trans)

    train_loader, test_loader = name_to_dict(
        name=args.dataset,
        arg=config,
        task=args.task,
    )

    print(f"Task: {args.task}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Loss: {args.loss}")
    print(f"Equivariant Transformation: {transform_dict['name']}")
    print(f"Noise Type: {args.noise_type}")
    if args.noise_type in ['gaussian', 'gaussian_poisson']:
        print(f"Gaussian Noise Sigma = {args.sigma}")
    if args.noise_type in ['poisson', 'gaussian_poisson']:
        print(f"Poisson Gain = {args.gain}")
    if args.task in ['sr', 'test_sr']:
        print(f"SR Data: {args.sr_data_name}, Factor: x{args.factor}")

    if args.task in ['sr', 'inpainting']:
        trainer = Trainer(
            task=args.task,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            alpha=args.alpha,
            factor=args.factor,
            ckpt_step=args.ckpt_step,
            standard='normal',
            sigma=args.sigma,
            index=args.index,
            mat_index=args.mat_index,
        )

        ckpt = None
        if args.ckpt_path and args.resume:
            ckpt = torch.load(args.ckpt_path, map_location=device)
            print(f"Resuming from checkpoint: {args.ckpt_path}")

        trainer.setup(
            model=model_dict,
            trainloader=train_loader,
            testloader=test_loader,
            physics=physics,
            transform=transform_dict,
            loss_type=args.loss,
            layers=args.layers,
            channel_dim=args.channel_dim,
            resume=args.resume,
            ckpt=ckpt,
            sr_data_name=args.sr_data_name,
            patch_size=args.patch_size,
            offset=args.offset,
            noise_type=args.noise_type,
            gain=args.gain
        )

        if args.task == "sr":
            print(f"Training Super Resolution with factor x{args.factor}")
            trainer.train_sr()
        elif args.task == 'inpainting':
            print("Training Inpainting")
            trainer.train_inpainting()

    elif args.task in ['test_sr', 'test_inpainting']:
        # Get checkpoint path
        ckpt_path = args.ckpt_path or get_checkpoint_path(args)
        print(f"Loading checkpoint: {ckpt_path}")

        sigma_test = args.sigma
        physics_test = physics

        tester = Tester(
            model=model_dict,
            device=device,
            task=args.task,
            physics=physics_test,
            ckpt_path=ckpt_path,
            sigma=sigma_test,
            factor=args.factor,
            standard='normal',
            load_physics=args.load_physics if hasattr(args, 'load_physics') else False
        )

        if args.task == "test_sr":
            tester.test_sr(test_loader=test_loader, sr_data_name=args.sr_data_name, patch_size=args.patch_size)

        elif args.task == "test_inpainting":
            tester.test_inpainting(test_loader, args.index, args.mat_index)


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    main()