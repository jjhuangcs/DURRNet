# DURRNet: Deep Unfolded Single Image Reflection Removal Network with Joint Prior
Jun-Jie Huang#(jjhuang@nudt.edu.cn), Tianrui Liu, Jingyuan Xia, Meng Wang, and Pier Luigi Dragotti

2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)


## Overview

- We propose a novel model-inspired and learning-based SIRR method called Deep Unfolded Reflection Removal Network (DURRNet). It combines the merits of both model-based and learning-based paradigms, leading to a more interpretable and effective deep architecture.
    
- We propose a model-based optimization approach and then obtain DURRNet by unfolding an iterative step into a Unfolded Separation Block (USB) based on proximal gradient descent. Key features of DURR-Net include the use of Invertible Neural Networks to impose the transform-based exclusion prior on the basis of natural image prior, as well as a coarse-to-fine architecture to fine-grain the reflection removal process.
  
- From extensive experimental results, the proposed DeMPAA method achieves a significantly higher attacking success rate when compared to other methods, and an imperceptible version of the proposed method i.e.,
DeMPAA-IP generates even more visually imperceptible adversarial patches to be practically feasible for attacking RSI scenes.

![Image text](./overview.png)


### Requisites

- PyTorch>=1.0
- Python>=3.7
- NVIDIA GPU + CUDA CuDNN

### Dataset

We follow the synthetic data generation model of CEILNet (https://openaccess.thecvf.com/content_iccv_2017/html/Fan_A_Generic_Deep_ICCV_2017_paper.html) and synthetic dataset (https://github.com/ceciliavision/perceptual-reflection-removal) contains 13700 pairs of indoor and outdoor images. The real datasets consist of Real89 (https://github.com/ceciliavision/perceptual-reflection-removal) which contains 89 aligned transmission and blended image pairs. All the datasets are publicly available.

### Key parameters
```--patch_size```: The total patch size. E.g. 0.01 ~= 1% of image.

```--patch_number```: The number of patches.

```--netClassifier```: The target classifier: resnet50/resnet34/resnet101/densenet121.

```--save_path```: The path to save the result.

```--data_path```: The data path.

```--train_size```: Number of training images.

```--test_size```: Number of test images.

```--max_count```: The max number of iterations to find adversarial example.

```--image_size```: The height / width of the input image to network.


### Run DeMPAA
We provide two versions of DeMPAA, i.e. DeMPAA and DeMPAA-IP, DeMPAA-IP is the more imperceptible version of DeMPAA.
- To Run *DeMPAA*:

```
python train_DeMPAA.py
```

- To Run *DeMPAA-IP*:

```
python train_DeMPAA_IP.py
```

### Result images
The visualization results of different adversarial patch attack methods on AID.
![Image text](./DeMPAA.png)

### Citation
```
@article{huang2024DeMPAA,
  title={DeMPAA: Deployable Multi-Mini-Patch Adversarial Attack for Remote Sensing Image Classification},
  author={Huang, Jun-Jie and Wang, Ziyue and Liu, Tianrui and Luo, Wenhan and Chen, Zihan and Zhao, Wentao and Wang, Meng},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  volume={62},
  pages={1-13},
  publisher={IEEE}
}

```
