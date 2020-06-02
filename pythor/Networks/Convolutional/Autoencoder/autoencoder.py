"""
Modules for building the manual CNN based autoencoder
"""
import os
from argparse import ArgumentParser

import torch
from torch import nn
from torch import optim
import torch.autograd as autograd 
from torch.autograd import Variable
from torch.nn import functional as F

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from pythor.datamodules import MNISTDataLoaders
from pythor.Networks.Convolutional.Autoencoder.autoencoder_components import Encoder, Decoder
from pythor.bots.botCallback import TelegramBotCallback
from pythor.bots.dl_bot import DLBot
from pythor.bots.config import telegram_config

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
    
class AutoEncoder(LightningModule):
    """
        CNN Autoencoder
    """
    def __init__(self, hparams=None):
        """
        CNN Autoencoder.
        Parameters to be included in hparams
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
        self.__check_hparams(hparams)
        self.hparams = hparams

        self.encoder = Encoder(self.input_shape,self.activation)
        self.decoder = Decoder(self.input_shape,self.activation)

        # NOTE Change dataloaders appropriately
        self.dataloaders = MNISTDataLoaders(save_path=os.getcwd())
        self.telegrad_logs = {} # log everything you want to be reported via telegram here

    def __check_hparams(self, hparams):
        self.input_shape = hparams.input_shape if hasattr(hparams,'input_shape') else (1,28,28)
        self.opt = hparams.opt if hasattr(hparams,'opt') else 'adam'
        self.batch_size = hparams.batch_size if hasattr(hparams,'batch_size') else 32
        self.lr = hparams.lr if hasattr(hparams,'lr') else 0.001
        self.weight_decay = hparams.weight_decay if hasattr(hparams,'weight_decay') else 0
        self.activation = hparams.activation if hasattr(hparams,'activation') else 'relu'
    
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def configure_optimizers(self):
        """
            Choose Optimizer
        """
        return optimizers[self.opt](self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def training_step(self, batch, batch_idx):
        """
            Define one training step
        """
        x, y = batch
        x_hat = self(x)  # get predictions from network
        criterion = nn.MSELoss()
        loss = criterion(x_hat,x)
        log = {'trainer_loss':loss}
        # self.logger.experiment.add_scalar('loss',loss)
        return {'loss': loss, 'log': log}
    
    def training_epoch_end(self, outputs):
        """
            Train Loss at the end of epoch
            Will store logs
        """
        avg_loss = torch.stack([x['trainer_loss'] for x in outputs]).mean()
        logs = {'trainer_loss_epoch': avg_loss}
        self.telegrad_logs['lr'] = self.lr # for telegram bot
        self.telegrad_logs['trainer_loss_epoch'] = avg_loss.item() # for telegram bot
        self.logger.log_metrics({'learning_rate':self.lr}) # if lr is changed by telegram bot
        return {'train_loss': avg_loss, 'log': logs}
  
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
            Validation at the end of epoch
            Will store logs
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss_epoch': avg_loss}
        self.telegrad_logs['val_loss_epoch'] = avg_loss.item() # for telegram bot
        return {'val_loss': avg_loss, 'log': logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        criterion = nn.MSELoss()
        x_hat = self(x)
        return {'test_loss': criterion(x_hat,x)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': logs}

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

    def val_dataloader(self):
        return self.dataloaders.val_dataloader(self.batch_size)

    def test_dataloader(self):
        return self.dataloaders.test_dataloader(self.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--input_shape', type=list, default=[1,28,28],
                            help='input image shape for MNIST')
        parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid', 'tanh'],
                            help='activations for nn layers')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='input vector shape for MNIST')
        # optimizer
        parser.add_argument('--opt', type=str, default='adam', choices=['adam', 'adamax', 'rmsprop'],
                            help='optimizer type for optimization')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='learning rate')
        parser.add_argument('--weight_decay', type=float, default=0,
                            help='weight decay in optimizer')
        return parser

def main():
    parser = ArgumentParser()
    # using this will log all params in mlflow board automatically
    parser = Trainer.add_argparse_args(parser) 
    parser = AutoEncoder.add_model_specific_args(parser)
    args = parser.parse_args()

    experiment_name = 'ConvAutoencoder'
    save_folder = 'model_weights/' + experiment_name
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    early_stopping = EarlyStopping('val_loss')
    # saves checkpoints to 'save_folder' whenever 'val_loss' has a new min
    checkpoint_callback = ModelCheckpoint(
                            filepath=save_folder+'/model_{epoch:02d}-{val_loss:.2f}')
    # tb_logger = loggers.TensorBoardLogger('logs')
    mlf_logger = MLFlowLogger(
                                experiment_name=experiment_name,
                                tracking_uri="file:./mlruns"
                                )

    # telegram
    token = telegram_config['token']
    user_id = telegram_config['user_id']
    bot = DLBot(token=token, user_id=user_id)
    telegramCallback = TelegramBotCallback(bot)

    model = AutoEncoder(args)

    trainer = Trainer(checkpoint_callback=checkpoint_callback,
                        early_stop_callback=early_stopping,
                        fast_dev_run=False,                     # make this as True only to check for bugs
                        max_epochs=1000,
                        resume_from_checkpoint=None,            # change this to model_path
                        logger=mlf_logger,                      # mlflow logger
                        callbacks=[telegramCallback],           # telegrad
                        )

    trainer.fit(model)
    trainer.test()

if __name__ == "__main__":
    main()