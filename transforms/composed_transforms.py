from __future__ import annotations
from deepinv.transform import Shift, Scale
from deepinv.transform.base import Transform
from transforms.spectral_transforms import SpectralScale


class InpaintingShiftScale(Transform):
    def __init__(self, device, spatial_shift_factors=None, spectral_factors=None, **kwargs):
        super().__init__(**kwargs)
        self.spatial = Shift(factors=spatial_shift_factors or [-2, -1, 1, 2])
        self.spectral = SpectralScale(factors=spectral_factors or [0.8, 0.9, 1.0, 1.1, 1.2], device=device)

    def _get_params(self, x):

        p1 = self.spatial._get_params(x)
        p2 = self.spectral._get_params(x)
        return {**p1, **p2}  

    def _transform(self, x, **kwargs):
        x = self.spatial._transform(x, **kwargs)
        x = self.spectral._transform(x, **kwargs)
        return x


class SRLowRes(Transform):
    def __init__(self, device, scale_factors=None, spectral_factors=None, **kwargs):
        super().__init__(**kwargs)
        self.spatial = Scale(factors=scale_factors or [0.5, 0.75], device=device)
        self.spectral = SpectralScale(factors=spectral_factors or [0.8, 0.9, 1.0, 1.1, 1.2], device=device)

    def _get_params(self, x):
        return {**self.spatial._get_params(x), **self.spectral._get_params(x)}

    def _transform(self, x, **kwargs):
        x = self.spatial._transform(x, **kwargs)
        x = self.spectral._transform(x, **kwargs)
        return x


def get_inpainting_transform(transform_type="shift_scale", **kwargs):

    transforms = {
        "shift_scale": InpaintingShiftScale(**kwargs),
    }
    if transform_type not in transforms:
        raise ValueError(f"Unknown transform_type: {transform_type}. Available: {list(transforms.keys())}")
    return transforms[transform_type]


def get_sr_transform(transform_type="scale", **kwargs):
  
    transforms = {
        "scale": SRLowRes(**kwargs),
    }
    if transform_type not in transforms:
        raise ValueError(f"Unknown transform_type: {transform_type}. Available: {list(transforms.keys())}")
    return transforms[transform_type]
