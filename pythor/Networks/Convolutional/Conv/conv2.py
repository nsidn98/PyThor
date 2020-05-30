"""
Modules for building the manual CNN
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


from pythor.datamodules import MNISTDataLoaders

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


class ConvNet(LightningModule):
    def __init__(self,  input_shape,
                        num_outputs,
                        activation='relu',
                        opt='adam',
                        batch_size=32,
                        lr=0.001,
                        weight_decay=0):
        super(ConvNet, self).__init__()
        """
        CNN followed by fully connected layers.
        Performs one 2x2 max pool after the first conv.
        Parameters
        ----------
        input_shape : tuple
            Dimension of input image cxlxb.
            Example: (3,210,150)
        num_outputs : int
            Dimension of output.
            Example: 10
        activation : str
            One of 'relu', 'sigmoid' or 'tanh' (the default is 'relu').
        opt : str
            One of 'adam' or 'adamax' or 'rmsprop' (defualt is 'adam')
        batch_size: int
            Batch size for training (default is 32)
        lr: float
            Learning rate for optimizer (default is 0.001)
        weight_decay: float
            Weight decay in optimizer (default is 0)
        """
        self.input_shape = input_shape
        self.num_outputs = num_outputs
        self.opt = opt
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        act = ACTS[activation]

        # NOTE Change dataloaders appropriately
        self.dataloaders = MNISTDataLoaders(save_path=os.getcwd())
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=4),
            act(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            act(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            act()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            act(),
            nn.Linear(512, self.num_outputs)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        """
            Get feature size after conv layers to flatten
        """
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def training_step(self, batch, batch_idx):
        """
            Define one training step
        """
        x, y = batch
        y_hat = self(x)  # get predictions from network
        loss = F.cross_entropy(y_hat, y)
        tensorboard_log = {'trainer_loss':loss}
        self.logger.experiment.add_scalar('loss',loss)
        return {'loss': loss, 'log': tensorboard_log}

    def configure_optimizers(self):
        """
            Choose Optimizer
        """
        return optimizers[self.opt](self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def prepare_data(self):
        """
            Prepare the dataset by downloading it 
            Will be run only for the first time if
            dataset is not available
        """
        self.dataloaders.prepare_data()

    def train_dataloader(self):
        """
            Refer dataset.py to make custom dataloaders
        """
        return self.dataloaders.train_dataloader(self.batch_size)

    def validation_step(self, batch, batch_idx):
        """
            One validation step
        """
        x, y = batch
        y_hat = self(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        """
            Validatio at the end of epoch
            Will store logs
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def val_dataloader(self):
        return self.dataloaders.val_dataloader(self.batch_size)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def test_dataloader(self):
        return self.dataloaders.test_dataloader(self.batch_size)


if __name__ == "__main__":
    save_folder = 'model_weights/'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    early_stopping = EarlyStopping('val_loss')
    # saves checkpoints to 'save_folder' whenever 'val_loss' has a new min
    checkpoint_callback = ModelCheckpoint(
                            filepath=save_folder+'model_{epoch:02d}-{val_loss:.2f}')
    tb_logger = loggers.TensorBoardLogger('logs/')

    model = ConvNet((1,28,28),10)
    trainer = Trainer(checkpoint_callback=checkpoint_callback,
                        early_stop_callback=early_stopping,
                        fast_dev_run=False,                      # make this as True only to check for bugs
                        max_epochs=1000,
                        resume_from_checkpoint=None,            # change this to model_path
                        logger=tb_logger,                       # tensorboard logger
                        )
    trainer.fit(model)
    trainer.test()