"""
Modules for building the CNN
Code inspired by @rajatvd
"""

"""
Modules for building the manual CNN
"""
import os

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers


from torchvision import datasets, transforms
from torchvision import transforms

from pythor.datamodules import MNISTDataLoaders

# %%
ACTS = {
    'relu':nn.ReLU,
    'sigmoid':nn.Sigmoid,
    'tanh':nn.Tanh,}

optimizers = {
    'adam': optim.Adam,
    'adamax': optim.Adamax,
    'rmsprop': optim.RMSprop,
    }

# %%

class ConvNet(LightningModule):
    """
    CNN for using 3x3 convs and one final FC layer.
    """
    def __init__(self,
                 input_size=28,
                 channels=None,
                 filters=None,
                 denses=None,
                 activation='relu',
                 opt='adam',
                 batch_size=32,
                 lr=0.001,
                 weight_decay=0):
        """CNN followed by fully connected layers.
        Performs one 2x2 max pool after the first conv.
        Parameters
        ----------
        input_size : int
            Dimension of input square image.
        channels : list of ints
            List of channels of conv layers including input channels
            (the default is [1,32,32,16,8]).
        filters : list of ints
            List of filter sizes for each of the conv layers
            Length of list should be one less than list of channels
            (the default is [3,3,3,3])
        denses : list of ints
            Sequence of linear layer outputs after the conv layers
            (the default is [10]).
        activation : str
            One of 'relu', 'sigmoid' or 'tanh' (the default is 'relu').
        """
        super().__init__()
        self.opt = opt
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        
        # NOTE Change dataloaders appropriately
        self.dataloaders = MNISTDataLoaders(save_path=os.getcwd())

        channels = channels or [1, 32, 32, 16, 8]
        filters = filters or [3, 3, 3, 3]
        denses = denses or [10]

        act = ACTS[activation]

        convs = [nn.Conv2d(kernel_size=k, in_channels=in_ch, out_channels=out_ch)
                 for in_ch, out_ch, k in zip(channels[:-1], channels[1:], filters)]

        if len(channels) <= 1:
            self.conv_net = None
            feature_count = input_size*input_size
        else:
            self.conv_net = nn.Sequential(
                convs[0],
                nn.MaxPool2d(kernel_size=2),
                act(),
                *[layer for tup in zip(convs[1:], [act() for _ in convs[1:]]) for layer in tup]
            )

            with torch.no_grad():
                test_inp = torch.randn(1, 1, input_size, input_size)
                features = self.conv_net(test_inp)
                feature_count = features.view(-1).shape[0]

        linears = [nn.Linear(in_f, out_f) for in_f, out_f in
                   zip([feature_count]+denses[:-1], denses)]

        self.dense = nn.Sequential(
            *[layer for tup in zip(linears, [act() for _ in linears]) for layer in tup][:-1]
        )


    def forward(self, input):
        if self.conv_net:
            input = self.conv_net(input)
        out = self.dense(input.view(input.shape[0], -1))
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)  # get predictions from network
        loss = F.cross_entropy(y_hat, y)
        tensorboard_log = {'trainer_loss':loss}
        self.logger.experiment.add_scalar('loss',loss)
        return {'loss': loss, 'log': tensorboard_log}

    def configure_optimizers(self):
        return optimizers[self.opt](self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def prepare_data(self):
        # download only
        self.dataloaders.prepare_data()

    '''
    def prepare_data(self):
        # stuff here is done once at the very beginning of training
        # before any distributed training starts

        # download stuff
        # save to disk
        # etc...
        ...

    def train_dataloader(self):
        # data transforms
        # dataset creation
        # return a DataLoader
        ...
    '''
    def train_dataloader(self):
        return self.dataloaders.train_dataloader(self.batch_size)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
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
# %%
if __name__ == "__main__":
    save_folder = 'model_weights/'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    early_stopping = EarlyStopping('val_loss')
    # saves checkpoints to 'save_folder' whenever 'val_loss' has a new min
    checkpoint_callback = ModelCheckpoint(
                            filepath=save_folder+'model_{epoch:02d}-{val_loss:.2f}')
    tb_logger = loggers.TensorBoardLogger('logs/')

    model = ConvNet(28)
    trainer = Trainer(checkpoint_callback=checkpoint_callback,
                        early_stop_callback=early_stopping,
                        fast_dev_run=False,                      # make this as True only to check for bugs
                        max_epochs=1000,
                        resume_from_checkpoint=None,            # change this to model_path
                        logger=tb_logger,                       # tensorboard logger
                        )
    trainer.fit(model)
    trainer.test()
