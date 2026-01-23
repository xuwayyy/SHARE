# SHARE: A Fully Unsupervised Framework for Single Hyperspectral Image Restoration

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2601.13987)
[![GitHub Stars](https://img.shields.io/github/stars/xuwayyy/SHARE?style=social)](https://github.com/xuwayyy/SHARE)


[SHARE: A Fully Unsupervised Framework for Single Hyperspectral Image Restoration]([https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Robust_Equivariant_Imaging_A_Fully_Unsupervised_Framework_for_Learning_To_CVPR_2022_paper.pdf](https://arxiv.org/pdf/2601.13987))
 
[Jiangwei Xie](https://scholar.google.co.uk/citations?user=gwCocBkAAAAJ&hl=en&oi=ao), [Zhang Wen](https://scholar.google.co.uk/citations?user=tzvIt4EAAAAJ&hl=en), [Mike E. Davies](https://www.research.ed.ac.uk/en/persons/michael-davies), [Dongdong Chen](https://dongdongchen.com)  

## Background

Hyperspectral image (HSI) restoration is a fundamental challenge in computational imaging and computer vision. It involves ill-posed inverse problems, such as inpainting and super-resolution. Although deep learning methods have transformed the field through data-driven learning, their effectiveness hinges on access to meticulously curated ground-truth datasets. This fundamentally restricts their applicability in real-world scenarios where such data is unavailable. This paper presents **SHARE** (Single Hyperspectral imAge Restoration with Equivariance), a **fully unsupervised** framework that unifies **geometric equivariance principles** with **low-rank spectral modelling** to eliminate the need for ground truth. SHARE's core concept is to exploit the intrinsic invariance of hyperspectral structures under differentiable geometric transformations (e.g. rotations and scaling) to derive self-supervision signals through equivariance consistency constraints. Our novel Dynamic Adaptive Spectral Attention (DASA) module further enhances this paradigm shift by explicitly encoding the global low-rank property of HSI and adaptively refining local spectral-spatial correlations through learnable attention mechanisms. Extensive experiments on HSI inpainting and super-resolution tasks demonstrate the effectiveness of SHARE. Our method outperforms many state-of-the-art unsupervised approaches and achieves performance comparable to that of supervised methods. We hope that our approach will shed new light on HSI restoration and broader scientific imaging scenarios.

---

## Environment Setup

Recommended **Python = 3.10** and **Pytorch=2.2.0** 

### Required Dependencies

```text
pytorch==2.2.0
deepinv==0.3.0
pyiqa==0.1.14.1
opencv-python==4.12.0
numpy==1.26.0
```
### Installation Example
```
conda create -n share python=3.10 -y
conda activate share
```
Install ```pytorch=2.2.0, cuda 12.1``` version from pytorch org first and install other essential resources
```
pip install deepinv==0.3.0
pip install pyiqa==0.1.14.1
pip install opencv-python==4.12.0
pip install numpy==1.26.0
```
## Dataset Preparation
For Chikusei Dataset, please download the Chikusei hyperspectral dataset from [Chikusei_Full_Image](https://naotoyokoya.com/Download.html), and download the 5 tiles for inpainting at [Chikusei_Test_5images.mat](https://drive.google.com/file/d/1hsE4uxQgHTZK-0amcCYIzFTAz5JRnipj/view?usp=share_link); Download PaviaU and Indian Pines dataset from [here](https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes#Indian_Pines)

After downloading, place the data under ```data/Matzoo```

## Training 

### Train Super-Resolution
Train Share with super-resolution on Cave dataset under $\times 2$ downsampling ratio, run
```
python main.py  --task sr --dataset Cave --factor 2 --sr_data_name fake_and_real_beers_ms.mat --lr 1e-3 --alpha 1 --transform Scale
```
Train Share with super-resolution on PaviaUni dataset under $\times 2$ downsamping ratio
```
python main.py  --task sr --dataset PaviaUni --factor 2 --lr 1e-3 --alpha 1 --transform Scale
```


### Train Inpainting
Train Share with inpainting on Chikusei dataset, run
```
python main.py --task inpainting --dataset Chikusei --lr 1e-2 --index your_index --mat_index your_mat_index --transform Shift
```
**index** means which tile from full Chikusei, **mat_index** means which mask shape

Train Share with inpainting on Indian Pines dataset, run
```
python main.py --task inpainting --dataset Indian -mat_index your_mat_index --lr 1e-2 --transform Shift-
```

### Other Features
We have provided a zoo of different transformations and loss functions, enjoy your trip.

## Testing
Similar with training command but configure your task from ```inpainting/sr``` to ```test_inpainting/test_inpainting``` , please make sure your testing params in command should align with training, e.g., lr should same.


## Citation
If you find our work useful, please cite:

```
@misc{xie2026sharefullyunsupervisedframework,
      title={SHARE: A Fully Unsupervised Framework for Single Hyperspectral Image Restoration},
      author={Jiangwei Xie and Zhang Wen and Mike Davies and Dongdong Chen},
      year={2026},
      eprint={2601.13987},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2601.13987},
}
```

## Acknowledgements
Most of our code is built upon [deepinverse](https://github.com/deepinv/deepinv), thanks for their framework
