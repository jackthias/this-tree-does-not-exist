# Final Project
## This Tree Does Not Exist

This project generates pictures of trees using Generative Adversarial
Networks (GANS). The neural networks used were built using PyTorch.
The dataset of tree images used to train the discriminator network was
the CIFAR100 Dataset, which was pulled into the project using
TorchVision.

The pictures generated, as well as the pictures used for training, are
3-color-channel 32x32 pictures. 

# Use

This project requires PyTorch and TorchVision in order to run.

```pip3 install torch torchvision```

Neural network parameters can be set at the top of `main.py`. 

Below the NN parameters, the `include_list` can be set. This is a list
of classes that the data loader will pull from the CIFAR100 dataset.
Currently, it is only set to pine trees. But I have also included the
commented out list of all trees.

The GAN can be run by executing `main.py` in a Python 3 interpreter.

Image output is at `/data/images/VIS_GAN/`.

CIFAR100 dataset must be obtained from CIFAR and is not distributed with this repo.

