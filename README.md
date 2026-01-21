# SHARE: A Fully Unsupervised Framework for Single Hyperspectral Image Restoration

📄 **Paper**:  
**SHARE: A Fully Unsupervised Framework for Single Hyperspectral Image Restoration**  
Jiangwei Xie\*, Zhang Wen\*, Mike Davies, Dongdong Chen  
\* Equal contribution  

📬 **Corresponding Author**:  
Dongdong Chen (d.chen@hw.ac.uk)
-Primary contact: Jiangwei Xie ([xiejiangweiouc@gmail.com](xiejiangweiouc@gmail.com))

📎 **arXiv**:  
https://arxiv.org/abs/2601.13987  

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
Install pytorch=2.2.0, cuda 12.1 version from github and install other essential resources
```
pip install deepinv==0.3.0
pip install pyiqa==0.1.14.1
pip install opencv-python==4.12.0
pip install numpy==1.26.0
```
## Dataset Preparation
For Chikusei Dataset, please download the Chikusei hyperspectral dataset from [Chikusei_Full_Image](https://naotoyokoya.com/Download.html), and download the 5 tiles for inpainting at [Chikusei_Test_5images.mat](https://drive.google.com/file/d/1hsE4uxQgHTZK-0amcCYIzFTAz5JRnipj/view?usp=share_link)

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
python main.py --task inpainting --dataset Chikusei --lr 1e-2 --index your_index --mat_index 0-3 --transform Shift
```
**index** means which tile from full Chikusei, **mat_index** means which mask shape

Train Share with inpainting on Indian Pines dataset, run
```
python main.py --task inpainting --dataset Indian --lr 1e-2 --transform Shift
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
