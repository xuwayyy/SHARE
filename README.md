# SHARE: A Fully Unsupervised Framework for Single Hyperspectral Image Restoration

📄 **Paper**:  
**SHARE: A Fully Unsupervised Framework for Single Hyperspectral Image Restoration**  
Jiangwei Xie\*, Zhang Wen\*, Mike Davies, Dongdong Chen  
\* Equal contribution  

📬 **Corresponding Author**:  
Dongdong Chen (d.chen@hw.ac.uk)

📎 **arXiv**:  
https://arxiv.org/abs/2601.13987  
https://arxiv.org/pdf/2601.13987

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

### Installation Example
```
conda create -n share python=3.10 -y
conda activate share

pip install deepinv==0.3.0
pip install pyiqa==0.1.14.1
pip install opencv-python==4.12.0
pip install numpy==1.26.0
```
## Dataset Preparation
For Chikusei Dataset, please download the Chikusei hyperspectral dataset from [Chikusei_Full_Image](https://naotoyokoya.com/Download.html), and download the 5 tiles for inpainting at [Chikusei_Test_5images.mat](https://drive.google.com/file/d/1hsE4uxQgHTZK-0amcCYIzFTAz5JRnipj/view?usp=share_link)

After downloading, place the data under ```data/Matzoo```


## Citation
If you find this work useful, please cite:
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
