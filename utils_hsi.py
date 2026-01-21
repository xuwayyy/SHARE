import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from data.Inpainting_data import get_inpainting_dataset
from data.Cave import CaveDataset, makeDataLoader
from deepinv.transform import Shift, Scale, Rotate, Reflect
from deepinv.transform.projective import Affine, Similarity, Euclidean, PanTiltRotate
from data.Pavia import makePaviaDataLoader
from data.Chikusei_SR import makeChikuseiDataLoader
import deepinv as dinv


def name_to_dict(name, arg, task, sr_mode="single"):
    loader = name_to_loader(name, arg, task, sr_mode=sr_mode)
    back_dict_train = loader_to_dict(loader, name)
    arg["mode"] = "test"
    loader = name_to_loader(name, arg, task, sr_mode=sr_mode)
    back_dict_test = loader_to_dict(loader, name)
    return back_dict_train, back_dict_test


def loader_to_dict(dataloader: [DataLoader, torch.Tensor], name: str):
    build_dict = {'name': name, 'data': dataloader}
    return build_dict


def name_to_loader(name: str, arg: dict, task, sr_mode='single'):

    mat_path = arg['mat_path']
    print(f"mat_path: {mat_path}")
    assert name in ['Cave', 'Indian', 'Chikusei', 'PaviaUni', 'Chikusei_SR'], f"{name} is not a valid name"
    print(f"arg sr data name: {arg['sr_data_name']}")
    if name == 'Cave':
        dataloader = makeDataLoader(mat_path=mat_path, task=sr_mode,
                                        mode=arg['mode'], transform=arg['transform'], name=arg['sr_data_name'])
    elif name == 'PaviaUni':
        dataloader = makePaviaDataLoader(mat_path=mat_path, transform=arg['transform'],
                                         patch_size=arg['patch_size'])
    elif name == 'Chikusei_SR':
        dataloader = makeChikuseiDataLoader(mat_path=mat_path, transform=arg['transform'],
                                         patch_size=arg['patch_size'], offset=arg['offset'])

    else:
        dataset_dict = get_inpainting_dataset(arg['device'], chikusei_index=arg['index'])
        chikusei, indian_pine = dataset_dict['chikusei'], dataset_dict['indian_pine']
        if name == 'Indian':
            dataloader = indian_pine
        else:
            dataloader = chikusei
    return dataloader


def transform_name_to_dict(name, n_trans):
    transform_list = ['Rotate', 'Shift', 'Scale', 'Reflect', 'Affine', 'Similarity', 'Euclidean', 'Tile']
    device = dinv.utils.get_freer_gpu()
    assert name in transform_list, f"{name} is not a valid name"
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
    else:
        raise NotImplementedError(f"{name} is not a valid transform name")
    back_dict = {'name': name, 'transform': ei}
    return back_dict
    