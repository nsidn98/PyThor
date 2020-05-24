"""
    Train Variational Autoencoder (Linear)
    Code taken from "@PyTorchLightning/pytorch-lightning-bolts"
"""
import os
from argparse import ArgumentParser

import torch
from torch import distributions
from torch.nn import functional as F

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers

from pythor.datamodules import MNISTDataLoaders
from pythor.Networks.Linear.Autoencoder.vae_components import Encoder, Decoder


class VAE(LightningModule):

    def __init__(
            self,
            hparams=None,
    ):
        super().__init__()
        # attach hparams to log hparams to the loggers (like tensorboard)
        self.__check_hparams(hparams)
        self.hparams = hparams

        self.dataloaders = MNISTDataLoaders(save_path=os.getcwd())

        self.encoder = self.init_encoder(self.input_dim, self.hidden_dim, self.latent_dim)
        self.decoder = self.init_decoder(self.input_dim, self.hidden_dim, self.latent_dim)

    def __check_hparams(self, hparams):
        self.input_dim = hparams.input_dim if hasattr(hparams, 'input_dim') else 28*28
        self.hidden_dim = hparams.hidden_dim if hasattr(hparams, 'hidden_dim') else 128
        self.latent_dim = hparams.latent_dim if hasattr(hparams, 'latent_dim') else 32
        self.batch_size = hparams.input_height if hasattr(hparams, 'batch_size') else 32

    def init_encoder(self, input_dim, hidden_dim, latent_dim):
        encoder = Encoder(input_dim, hidden_dim, latent_dim)
        return encoder

    def init_decoder(self, input_dim, hidden_dim, latent_dim):
        decoder = Decoder(input_dim, hidden_dim, latent_dim)
        return decoder

    def get_prior(self, z_mu, z_std):
        # Prior ~ Normal(0,1)
        P = distributions.normal.Normal(loc=torch.zeros_like(z_mu), scale=torch.ones_like(z_std))
        return P

    def get_approx_posterior(self, z_mu, z_std):
        # Approx Posterior ~ Normal(mu, sigma)
        Q = distributions.normal.Normal(loc=z_mu, scale=z_std)
        return Q

    def elbo_loss(self, x, P, Q):
        # Reconstruction loss
        z = Q.rsample()
        pxz = self(z)
        recon_loss = F.binary_cross_entropy(pxz, x, reduction='none')

        # sum across dimensions because sum of log probabilities of iid univariate gaussians is the same as
        # multivariate gaussian
        recon_loss = recon_loss.sum(dim=-1)

        # KL divergence loss
        log_qz = Q.log_prob(z)
        log_pz = P.log_prob(z)
        kl_div = (log_qz - log_pz).sum(dim=1)

        # ELBO = reconstruction + KL
        loss = recon_loss + kl_div

        # average over batch
        loss = loss.mean()
        recon_loss = recon_loss.mean()
        kl_div = kl_div.mean()

        return loss, recon_loss, kl_div, pxz

    def forward(self, z):
        return self.decoder(z)

    def _run_step(self, batch):
        x, _ = batch
        # NOTE comment the line below, just for testing MNIST
        x = x.view(-1, self.input_dim)
        z_mu, z_log_var = self.encoder(x)
        z_std = torch.exp(z_log_var / 2)

        P = self.get_prior(z_mu, z_std)
        Q = self.get_approx_posterior(z_mu, z_std)

        x = x.view(x.size(0), -1)

        loss, recon_loss, kl_div, pxz = self.elbo_loss(x, P, Q)

        return loss, recon_loss, kl_div, pxz

    def training_step(self, batch, batch_idx):
        loss, recon_loss, kl_div, pxz = self._run_step(batch)

        tensorboard_logs = {
            'train_elbo_loss': loss,
            'train_recon_loss': recon_loss,
            'train_kl_loss': kl_div
        }

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, recon_loss, kl_div, pxz = self._run_step(batch)

        return {
            'val_loss': loss,
            'val_recon_loss': recon_loss,
            'val_kl_div': kl_div,
            'pxz': pxz
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        recon_loss = torch.stack([x['val_recon_loss'] for x in outputs]).mean()
        kl_loss = torch.stack([x['val_kl_div'] for x in outputs]).mean()

        tensorboard_logs = {'val_elbo_loss': avg_loss,
                            'val_recon_loss': recon_loss,
                            'val_kl_loss': kl_loss}

        return {
            'avg_val_loss': avg_loss,
            'log': tensorboard_logs
        }

    def test_step(self, batch, batch_idx):
        loss, recon_loss, kl_div, pxz = self._run_step(batch)

        return {
            'test_loss': loss,
            'test_recon_loss': recon_loss,
            'test_kl_div': kl_div,
            'pxz': pxz
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        recon_loss = torch.stack([x['test_recon_loss'] for x in outputs]).mean()
        kl_loss = torch.stack([x['test_kl_div'] for x in outputs]).mean()

        tensorboard_logs = {'test_elbo_loss': avg_loss,
                            'test_recon_loss': recon_loss,
                            'test_kl_loss': kl_loss}

        return {
            'avg_test_loss': avg_loss,
            'log': tensorboard_logs
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def prepare_data(self):
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
        parser.add_argument('--hidden_dim', type=int, default=128,
                            help='itermediate layers dimension before embedding for default encoder/decoder')
        parser.add_argument('--latent_dim', type=int, default=32,
                            help='dimension of latent variables z')
        parser.add_argument('--input_width', type=int, default=28,
                            help='input image width - 28 for MNIST (must be even)')
        parser.add_argument('--input_height', type=int, default=28,
                            help='input image height - 28 for MNIST (must be even)')
        parser.add_argument('--batch_size', type=int, default=32)
        return parser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = VAE.add_model_specific_args(parser)
    args = parser.parse_args()

    save_folder = 'model_weights/'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    early_stopping = EarlyStopping('avg_val_loss')
    # saves checkpoints to 'save_folder' whenever 'val_loss' has a new min
    checkpoint_callback = ModelCheckpoint(
                            filepath=save_folder+'model_{epoch:02d}-{val_loss:.2f}')
    tb_logger = loggers.TensorBoardLogger('logs/')


    vae = VAE(args)
    trainer = Trainer(checkpoint_callback=checkpoint_callback,
                        early_stop_callback=early_stopping,
                        fast_dev_run=False,                      # make this as True only to check for bugs
                        max_epochs=1000,
                        resume_from_checkpoint=None,            # change this to model_path
                        logger=tb_logger,                       # tensorboard logger
                        )
    trainer.fit(vae)
    trainer.test()