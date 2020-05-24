"""
Modules for training a classifier with a pretrained 
autoencode feature extractor

NOTE : Under construction
Not tested 
"""
import os

import torch
from torch import nn
from torch import optim
import torch.autograd as autograd 
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers


from torchvision.datasets import CIFAR10
from torchvision import datasets, transforms
from torchvision import transforms

from autoencoder import Encoder, Decoder, AutoEncoder

# %%
ACTS = {
    'relu':nn.ReLU,
    'sigmoid':nn.Sigmoid,
    'tanh':nn.Tanh,
    }

optimizers = {
    'adam': optim.Adam,
    'adamax': optim.Adamax,
    'rmsprop': optim.RMSprop,
    }

class CIFARClassifier(LightningModule):
    """
        Classifier with pre-trained autoencoder as feature extractor
    """
    def __init__(self,):
        # init the pretrained LightningModule
        self.feature_extractor = AutoEncoder.load_from_checkpoint(PATH)
        self.feature_extractor.freeze()

        # NOTE: the autoencoder outputs a 100-dim representation and CIFAR-10 has 10 classes
        # Change this in the autoencoder.py appropriately
        # This class is just an example for preparing pre-training networks
        self.classifier = nn.Linear(100, 10)

    def forward(self, x):
        representations = self.feature_extractor(x)
        x = self.classifier(representations)
        return x

# %%
if __name__ == "__main__":
    model = CIFARClassifier()
    trainer = Trainer()
    trainer.fit(model)

    # test 
    model = CIFARClassifier.load_from_checkpoint(PATH)
    model.freeze()

    # TODO: Implement some_images_from_cifar10()
    x = some_images_from_cifar10()
    predictions = model(x)
