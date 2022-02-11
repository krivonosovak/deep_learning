# DCGAN and VAE pytorch implementation  

-----------

Get `tensorboard_data` folder here: [link](https://www.dropbox.com/s/mkvcybytdxllo76/tensorboard_data.zip?dl=0)

-----------
## DCGAN
DCGAN paper:  [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1511.06434)

![](images/GANs.png)

### Implementation details
from [github](https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN/)
![](images/pytorch_DCGAN.png) 

### Train dataset 

Training on [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html):

![](images/cifar_images.png)

### Result:
Generating examples after training:

![](images/individualImage.png)

![](images/cifar_viz.gif)


## VAE

![](images/autoencoder_schema.jpg)

### Train dataset 

Training on FashionMNIST:

![](images/fashion-mnist-sprite.png)


### Result:

Latent space representation after training: 

![](images/viz_dist.png) ![](images/distribution_viz.gif)
