import torch
from torch.utils.data import DataLoader
import deepinv as dinv

from data.Inpainting_data import get_inpainting_dataset
from data.Cave import makeDataLoader
from data.Pavia import makePaviaDataLoader
from data.heipor import makeHeiPorDataloader
from data.agrifood import makeAgriFoodDataloader
from data.braintumor import makeBrainDataloader

from deepinv.transform import Shift, Scale, Rotate, Reflect
from deepinv.transform.projective import Affine, Similarity, Euclidean, PanTiltRotate
from transforms.composed_transforms import InpaintingShiftScale, SRLowRes
from transforms.spectral_transforms import SpectralScale




def name_to_dict(name, arg, task):
    loader = name_to_loader(name, arg, task)
    back_dict_train = loader_to_dict(loader, name)
    arg["mode"] = "test"
    loader = name_to_loader(name, arg, task)
    back_dict_test = loader_to_dict(loader, name)
    return back_dict_train, back_dict_test


def loader_to_dict(dataloader: [DataLoader, torch.Tensor], name: str):
    build_dict = {'name': name, 'data': dataloader}
    return build_dict


def name_to_loader(name: str, arg: dict, task, ):
    if task not in ['sr_real', 'test_sr_real']:
        mat_path = arg['mat_path']
    else:
        mat_path = arg['sr_real_mat_path']

    retain_ratio = arg.get('retain_ratio', 1.0)    # ← 取出

    if name == 'Cave':
        dataloader = makeDataLoader(mat_path=mat_path, mode=arg['mode'],
                                    transform=arg['transform'], name=arg['sr_data_name'], retain_ratio=retain_ratio)
    elif name == 'PaviaUni':
        dataloader = makePaviaDataLoader(mat_path=mat_path, transform=arg['transform'],
                                         patch_size=arg['patch_size'], retain_ratio=retain_ratio)
    elif name == 'HeiPor':
        dataloader = makeHeiPorDataloader(mat_path=mat_path, transform=arg['transform'],
                                          patch_size=arg['patch_size'], retain_ratio=retain_ratio)
    elif name in ['AgriFood', 'AgriFood2']:
        dataloader, wavelength = makeAgriFoodDataloader(mat_path=mat_path, transform=arg['transform'], retain_ratio=retain_ratio,
                                            patch_size=arg['patch_size'], band_division=4)
    elif name in ['Brain1', 'Brain2']:
        dataloader = makeBrainDataloader(hdr_path=mat_path, transform=arg['transform'], retain_ratio=retain_ratio,
                                         patch_size=arg['patch_size'], band_division=10)
    else:
        dataset_dict = get_inpainting_dataset(arg['device'], chikusei_index=arg['index'], retain_ratio=retain_ratio,)
        chikusei, indian_pine = dataset_dict['chikusei'], dataset_dict['indian_pine']
        if name == 'Indian':
            dataloader = indian_pine
        else:
            dataloader = chikusei
    return dataloader


def transform_name_to_dict(name, n_trans):
    # transform_list = ['Rotate', 'Shift', 'Scale', 'Reflect', 'Affine', 'Similarity', 'Euclidean', 'Tile',
    #                   'ScaleScale', 'ShiftScaleScale']
    # assert name in transform_list, f"{name} is not a valid name"
    device = dinv.utils.get_freer_gpu()
    if name == 'Rotate':
        ei = Rotate(n_trans=n_trans)
    elif name == 'Shift':
        ei = Shift(n_trans=n_trans)
    elif name == 'Scale':
        ei = Scale(n_trans=n_trans)
    elif name == 'Affine':
        ei = Affine(n_trans=n_trans, device=device)
    elif name == 'Reflect':
        ei = Reflect(n_trans=n_trans, device=device)
    elif name == 'Similarity':
        ei = Similarity(n_trans=n_trans, device=device)
    elif name == 'Euclidean':
        ei = Euclidean(n_trans=n_trans, device=device)
    elif name == 'Tile':
        ei = PanTiltRotate(n_trans=n_trans, device=device)
    elif name == 'InpaintingShiftScale':
        ei = InpaintingShiftScale(device=device)
    elif name == 'SpectralScale':
        ei = SpectralScale(device=device)
    elif name == 'ScaleScale':
        ei = SRLowRes(device=device)
    else:
        raise NotImplementedError(f"{name} is not a valid transform name")
    back_dict = {'name': name, 'transform': ei}
    return back_dict

