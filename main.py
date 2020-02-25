"""
Jack Thias 2019
This code implements a Visual GAN that creates images of trees based on the trees included in the CIFAR100 Dataset.
PyTorch & TorchVision are used to grab the dataset and to implement the neural networks.

This code was created with the assistance of these guides:
* https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
* https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/generative/dcgan.ipynb
* https://stackoverflow.com/questions/54380140/how-do-i-extract-only-subset-of-classes-from-torchvision-datasets-cifar10
*
"""
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import datasets
from torchvision.transforms import transforms

from utils import Logger

BATCH_SIZE = 100
DROPOUT_P = 0.3
NEGATIVE_SLOPE = 0.2
INPUT_SIZE = 32 * 32 * 3  # 32x32 image with 3 color channels
GENERATOR_FEATURES = BATCH_SIZE
OPTIMIZER_LEARNING_RATE = 0.0002
TEST_NOISE_SAMPLES = 16
NUM_EPOCHS = 20000

# include_list = [47, 52, 56, 59, 96]
include_list = [59]

"""
Designed with help from:
https://stackoverflow.com/questions/54380140/how-do-i-extract-only-subset-of-classes-from-torchvision-datasets-cifar10
"""
class SubLoader(datasets.CIFAR100):
    def __init__(self, *args, exclude_list=[], **kwargs):
        super(SubLoader, self).__init__(*args, **kwargs)

        if exclude_list == []:
            return

        self.data = [img for img, label in zip(self.data, self.targets) if label not in exclude_list]


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.hidden0 = nn.Sequential(
            nn.Linear(INPUT_SIZE, 2048),
            nn.LeakyReLU(NEGATIVE_SLOPE),
            nn.Dropout(DROPOUT_P)
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LeakyReLU(NEGATIVE_SLOPE),
            nn.Dropout(DROPOUT_P),
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(NEGATIVE_SLOPE),
            nn.Dropout(DROPOUT_P)
        )

        self.output = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)

        return self.output(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.hidden0 = nn.Sequential(
            nn.Linear(GENERATOR_FEATURES, 256),
            nn.LeakyReLU(NEGATIVE_SLOPE)
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(NEGATIVE_SLOPE)
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(NEGATIVE_SLOPE)
        )

        self.out = nn.Sequential(
            nn.Linear(1024, INPUT_SIZE),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)

        return self.out(x)


def images_to_vectors(images): return images.view(images.size(0), INPUT_SIZE)


def vectors_to_images(vectors): return vectors.view(vectors.size(0), 3, 32, 32)


def noise(size): return Variable(torch.randn(size, BATCH_SIZE))


def ones_target(size): return Variable(torch.ones(size, 1))


def zeroes_target(size): return Variable(torch.zeros(size, 1))


def train_discriminator(optimizer, real_data, fake_data):
    def train(data, real):
        prediction = discriminator(data)
        error = loss(prediction, ones_target(n) if real else zeroes_target(n))
        error.backward()
        return prediction, error

    n = real_data.size(0)
    optimizer.zero_grad()
    prediction_real, error_real = train(real_data, True)
    prediction_fake, error_fake = train(fake_data, False)
    optimizer.step()

    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer, fake_data):
    n = fake_data.size(0)
    optimizer.zero_grad()
    prediction = discriminator(fake_data)
    error = loss(prediction, ones_target(n))
    error.backward()
    optimizer.step()
    return error


# Load in the CIFAR100 Dataset
exclude_list = [x for x in range(0,100) if x not in include_list]

ds = SubLoader(".", train=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((.5, .5, .5), (.5, .5, .5))
                       ]),
                       download=True, exclude_list=exclude_list)
data_loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
num_batches = len(data_loader)

discriminator = Discriminator()
generator = Generator()
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=OPTIMIZER_LEARNING_RATE)
generator_optimizer = optim.Adam(generator.parameters(), lr=OPTIMIZER_LEARNING_RATE)

loss = nn.BCELoss()
test_noise = noise(TEST_NOISE_SAMPLES)
logger = Logger(model_name='VIS_GAN', data_name='PineTrees-CIFAR100')

for epoch in range(NUM_EPOCHS):
    for batch_num, (real_batch, _) in enumerate(data_loader):
        n = real_batch.size(0)

        real_data = Variable(images_to_vectors(real_batch))
        # logger.log_images(real_batch, TEST_NOISE_SAMPLES, epoch, batch_num, num_batches)
        fake_data = generator(noise(n)).detach()
        discriminator_error, discriminator_real_prediction, discriminator_fake_prediction = \
            train_discriminator(discriminator_optimizer, real_data, fake_data)

        fake_data = generator(noise(n))
        generator_error = train_generator(generator_optimizer, fake_data)
        logger.log(discriminator_error, generator_error, epoch, batch_num, num_batches)

        if (batch_num % BATCH_SIZE) == 0:
            test_images = vectors_to_images(generator(test_noise))
            test_images = test_images.data

            logger.log_images(test_images, TEST_NOISE_SAMPLES, epoch, batch_num, num_batches)
            logger.display_status(epoch, NUM_EPOCHS, batch_num, num_batches, discriminator_error, generator_error,
                                  discriminator_real_prediction, discriminator_fake_prediction)
