"""
Modules for building the CNN
Code inspired by @rajatvd
"""
import os
from argparse import ArgumentParser

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from pythor.datamodules import MNISTDataLoaders
from pythor.bots.botCallback import TelegramBotCallback
from pythor.bots.dl_bot import DLBot
from pythor.bots.config import telegram_config


ACTS = {
    'relu':nn.ReLU,
    'sigmoid':nn.Sigmoid,
    'tanh':nn.Tanh,}

optimizers = {
    'adam': optim.Adam,
    'adamax': optim.Adamax,
    'rmsprop': optim.RMSprop,
    }

class ConvNet(LightningModule):
    """
    CNN for using 3x3 convs and one final FC layer.
    """
    def __init__(self, hparams = None):
        """CNN followed by fully connected layers.
        Performs one 2x2 max pool after the first conv.
        Parameters to be included in hparams
        ----------
        input_shape : int
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
        opt : str
            One of 'adam' or 'adamax' or 'rmsprop'.
            Default : 'adam'
        batch_size: int
            Batch size for training.
            Default : 32
        lr: float
            Learning rate for optimizer.
            Default : 0.001
        weight_decay: float
            Weight decay in optimizer.
            Default : 0
        """
        super().__init__()
        self.__check_hparams(hparams)
        self.hparams = hparams

        # NOTE Change dataloaders appropriately
        self.dataloaders = MNISTDataLoaders(save_path=os.getcwd())
        self.telegrad_logs = {} # log everything you want to be reported via telegram here

        convs = [nn.Conv2d(kernel_size=k, in_channels=in_ch, out_channels=out_ch)
                 for in_ch, out_ch, k in zip(self.channels[:-1], self.channels[1:], self.filters)]

        if len(self.channels) <= 1:
            self.conv_net = None
            feature_count = self.input_shape*self.input_shape
        else:
            self.conv_net = nn.Sequential(
                convs[0],
                nn.MaxPool2d(kernel_size=2),
                self.act(),
                *[layer for tup in zip(convs[1:], [self.act() for _ in convs[1:]]) for layer in tup]
            )

            with torch.no_grad():
                test_inp = torch.randn(1, 1, self.input_shape, self.input_shape)
                features = self.conv_net(test_inp)
                feature_count = features.view(-1).shape[0]

        linears = [nn.Linear(in_f, out_f) for in_f, out_f in
                   zip([feature_count]+self.denses[:-1], self.denses)]

        self.dense = nn.Sequential(
            *[layer for tup in zip(linears, [self.act() for _ in linears]) for layer in tup][:-1]
        )

    def __check_hparams(self, hparams):
        self.input_shape = hparams.input_shape if hasattr(hparams,'input_shape') else 28
        self.channels = hparams.channels if hasattr(hparams, 'channels') else [1, 32, 32, 16, 8]
        self.filters = hparams.filters if hasattr(hparams, 'filters') else [3, 3, 3, 3]
        self.denses = hparams.denses if hasattr(hparams, 'denses') else [10]
        self.activation = hparams.activation if hasattr(hparams,'activation') else 'relu'
        self.opt = hparams.opt if hasattr(hparams,'opt') else 'adam'
        self.batch_size = hparams.batch_size if hasattr(hparams,'batch_size') else 32
        self.lr = hparams.lr if hasattr(hparams,'lr') else 0.001
        self.weight_decay = hparams.weight_decay if hasattr(hparams,'weight_decay') else 0
        self.act = ACTS[self.activation]

    def forward(self, input):
        if self.conv_net:
            input = self.conv_net(input)
        out = self.dense(input.view(input.shape[0], -1))
        return out

    def configure_optimizers(self):
        return optimizers[self.opt](self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)  # get predictions from network
        loss = F.cross_entropy(y_hat, y)
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
        x, y = batch
        y_hat = self(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss_epoch': avg_loss}
        self.telegrad_logs['val_loss_epoch'] = avg_loss.item() # for telegram bot
        return {'val_loss': avg_loss, 'log': logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss_epoch': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': logs}

    def prepare_data(self):
        # download only
        self.dataloaders.prepare_data()

    def train_dataloader(self):
        return self.dataloaders.train_dataloader(self.batch_size)

    def val_dataloader(self):
        return self.dataloaders.val_dataloader(self.batch_size)

    def test_dataloader(self):
        return self.dataloaders.test_dataloader(self.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--input_shape', type=int, default=28,
                            help='input image dim for MNIST (must be square image)')
        parser.add_argument('--channels', type=list, default=[1, 32, 32, 16, 8],
                            help='List of channels in each conv layer including input')
        parser.add_argument('--filters', type=list, default=[3, 3, 3, 3],
                            help='List of filter sizes for each of the conv layers. Length of list should be one less than list of channels')
        parser.add_argument('--denses', type=list, default=[10],
                            help='List of linear layer outputs after the conv layers')
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
    parser = ConvNet.add_model_specific_args(parser)
    args = parser.parse_args()

    experiment_name = 'conv1'
    # tb_logger = loggers.TensorBoardLogger('logs')
    mlf_logger = MLFlowLogger(
                                experiment_name=experiment_name,
                                tracking_uri="file:./mlruns"
                                )
    save_folder = 'model_weights/' + experiment_name + '/'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    save_folder = save_folder + mlf_logger.run_id + '/'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    early_stopping = EarlyStopping('val_loss')
    # saves checkpoints to 'save_folder' whenever 'val_loss' has a new min
    checkpoint_callback = ModelCheckpoint(
                            filepath=save_folder+'/model_{epoch:02d}-{val_loss:.2f}')

    # telegram
    token = telegram_config['token']
    user_id = telegram_config['user_id']
    bot = DLBot(token=token, user_id=user_id)
    telegramCallback = TelegramBotCallback(bot)

    model = ConvNet(args)

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