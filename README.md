# Generalized-Zero-Shot-Learning-via-Aligned-Variational-Autoencoders

This repository has the pytorch implementation of the paper "Generalized Zero-and Few-Shot Learning via Aligned Variational Autoencoders." (CVPR 2019) [[pdf]](https://arxiv.org/pdf/1812.01784.pdf)

This repository has the implementation of zero-shot learning in a Generalized setting and has been tested on 4 datasets.

Note: I am still working on improving the results.

## Dataset

The dataset splits can be downloaded [here](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/), please download the `Proposed Split` and place it in the same folder. 

Find additional details about the dataset in the `README.md` of the `Proposed split`.

## Training and Testing

Download the pretrained model for various datasets [[here]]() and place it in `models/`

1. For Testing:

```
python linear_classifier.py --dataset CUB --dataset_path xlsa17/data/CUB/ --model_path models/checkpoint_cada_CUB.pth --pretrained
```
Change the arguments according to the dataset

2. For Training:
```
python linear_classifier.py --dataset CUB --dataset_path xlsa17/data/CUB/

```

## Results

| Dataset | Paper Results </br> (s, u, h) | Respository Results </br> (s, u, h) |
|--|--|--|
| CUB| 53.5, 51.6, 52.4 | 53.52, 47.29, 50.21 |
|AWA1| 72.8, 57.3, 64.1 | 73.54, 46.69, 57.19 |
|AWA2| 75.0, 55.8, 63.9 | 82.77, 44.94, 58.25 |
|SUN| 35.7, 47.2, 40.6 | |

