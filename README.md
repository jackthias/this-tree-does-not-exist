# Final Project from COMP251: Machine Learning
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

# Progression

At epoch 0, the Generator is merely generating random noise:
![Epoch 0](https://github.com/jackthias/this-tree-does-not-exist/blob/master/examples/_epoch_0_batch_0.png)

Very quickly (epoch 3), the adjust to the general structure:
![Epoch 3](https://github.com/jackthias/this-tree-does-not-exist/blob/master/examples/_epoch_3_batch_0.png)

Overtime (epoch 100) colors become less noisy:
![Epoch 100](https://github.com/jackthias/this-tree-does-not-exist/blob/master/examples/_epoch_100_batch_0.png)

Eventually (epoch 500), we start to see some variety come out of the generator.
![Epoch 500](https://github.com/jackthias/this-tree-does-not-exist/blob/master/examples/_epoch_500_batch_0.png)

# Conclusions

I am impressed by how quickly the generator is able to form the general structure of the tree. Over more epochs (around 10,000), though, the results start to regress and overfit to the classifiers model of a tree. Resulting in images that are less tree-like than even earlier epochs. I suspect that this is because of the limited dataset used.
