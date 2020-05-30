"""
Modules for building the manual CNN based autoencoder
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

class Encoder(torch.nn.Module):
    """
        Encoder part for the autoencoder
    """
    def __init__(self, input_shape,
                        activation='relu',
                        ):
        """
            Parameters:
            -----------
            input_shape : tuple
                Tuple of input shape of image cxhxw
                Example : (3,28,28)
            activation : str
                One of 'relu', 'sigmoid' or 'tanh' (the default is 'relu').
        """
        super(Encoder, self).__init__()
        act = ACTS[activation]
        self.input_shape = input_shape

        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 6, kernel_size=5),
            act(inplace=True),
            nn.Conv2d(6,16,kernel_size=5),
            act(inplace=True))
        
    def forward(self,x):
        return self.encoder(x)
    
    def feature_size(self):
        return self.encoder(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

class Decoder(torch.nn.Module):
    """
        Decoder part for the autoencoder
    """
    def __init__(self, input_shape,
                       activation='relu',
                        ):
        """
            Parameters:
            -----------
            input_shape : tuple
                Tuple of input shape of image cxhxw
                Example : (3,28,28)
            
            activation : str
                One of 'relu', 'sigmoid' or 'tanh' (the default is 'relu').
        """
        super(Decoder, self).__init__()
        act = ACTS[activation]
        self.decoder = nn.Sequential(             
                nn.ConvTranspose2d(16,6,kernel_size=5),
                act(inplace=True),
                nn.ConvTranspose2d(6,input_shape[0],kernel_size=5),
                act(inplace=True))
    
    def forward(self,x):
        return self.decoder(x)

class AutoEncoder(LightningModule):
    """
        CNN Autoencoder
    """
    def __init__(self,input_shape,
                        activation='relu',
                        opt='adam',
                        batch_size=32,
                        lr=0.001,
                        weight_decay=1e-5):
        """
        CNN Autoencoder.
        Parameters
        ----------
        input_shape : int
            Dimension of input image cxlxb.
            Example: [3,210,150]
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
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_shape,activation)
        self.decoder = Decoder(input_shape,activation)

        # NOTE Change dataloaders appropriately
        self.dataloaders = MNISTDataLoaders(save_path=os.getcwd())

        self.opt = opt
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        act = ACTS[activation]
    
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        """
            Define one training step
        """
        x, y = batch
        x_hat = self(x)  # get predictions from network
        criterion = nn.MSELoss()
        loss = criterion(x_hat,x)
        tensorboard_log = {'trainer_loss':loss}
        # self.logger.experiment.add_scalar('loss',loss)
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
            Refer dataset.py and pythor.datamodules
            to make custom dataloaders
        """
        return self.dataloaders.train_dataloader(self.batch_size)
  

    def validation_step(self, batch, batch_idx):
        """
            One validation step
        """
        x, y = batch
        criterion = nn.MSELoss()
        x_hat = self(x)
        return {'val_loss': criterion(x_hat,x)}

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
        criterion = nn.MSELoss()
        x_hat = self(x)
        return {'test_loss': criterion(x_hat,x)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        print()
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

    model = AutoEncoder((1,28,28))
    trainer = Trainer(checkpoint_callback=checkpoint_callback,
                        early_stop_callback=early_stopping,
                        fast_dev_run=False,                      # make this as True only to check for bugs (will run one epoch)
                        max_epochs=1000,
                        resume_from_checkpoint=None,            # change this to model_path
                        logger=tb_logger,                       # tensorboard logger
                        )
    trainer.fit(model)
    trainer.test()