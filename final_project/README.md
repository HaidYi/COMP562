# General Purpose Image Generator by WGAN
Autohrs: Jingyuan Chen, Tenghao Huang, Jiyao Wang, Haidong Yi

## Abstract
In this project, we reproduced the implementation of Wasserstein Generative Adversarial Network (WGAN) and trained it on several datasets for image generation purpose. The datasets include multi-class natural photos, human face images, and animation character images. Results are measured by Frechet Inception Distance (FID).
## Introduction
Image generation is always a field of interest in machine learning. The Generative Adversarial Network (GAN) is proved to be an effective model in generating plausible fake images from training sets. Related work concerning this subject has introduced several variations of this model including DCGAN, CycleGAN, and WGAN. Our project aims at examining the power of WGAN in image generation, especially its application in generating human designed images.
## Methods
A Generative Adversarial Network (GAN) is a deep learning generative model including a discriminator and a generator. The generator G tries to generate fake images to fool the discriminator while the discriminator D tries to classify fake and real images. <br>
<br>
In our implementation, the generator has a total of 4 deconvolutional layers implemented with transpose convolution, each has a filter size of 4 and stride of 2 except for the first. We added batch norm and ReLU layers after each layer except the last. The discriminator has a total of 4 convolutional layers, and we added instance norm and leaky ReLU layers after each except for the last. We use no fully connected layers and skip the log operation on the output.
## Dependencies 
* numpy
* pytorch (>=1.3)
* torchvision
* tqdm
* pyyaml
* tensorboardX

## Usage



## Reference
This repo partially referenced the implementation of W-GAN in this [repo](https://github.com/Zeleni9/pytorch-wgan).

Arjovsky, Martin, Soumith Chintala, and Léon Bottou. "Wasserstein generative adversarial networks." International conference on machine learning. 2017.

Gulrajani, Ishaan, et al. "Improved training of wasserstein gans." Advances in neural information processing systems. 2017.
