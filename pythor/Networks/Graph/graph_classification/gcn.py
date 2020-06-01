"""
    Graph Convolutional Network (GCN) for graph classification
    Graph will be classified to one of the classes
    Example provided for covid drug design challenge by MIT aicures.
"""
import os
import numpy as np
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv, NNConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from pythor.datamodules.molecule_dataloader import MoleculeDataloaders
from pythor.bots.botCallback import TelegramBotCallback
from pythor.bots.dl_bot import DLBot
from pythor.bots.config import telegram_config


POOL = {
    'mean': global_mean_pool,
    'add' : global_add_pool,
    'max' : global_max_pool,
}

ACTS = {
    'relu':F.relu,
    'sigmoid':nn.Sigmoid,
    'tanh':nn.Tanh,
    }

optimizers = {
    'adam': optim.Adam,
    'adamax': optim.Adamax,
    'rmsprop': optim.RMSprop,
    }

class GCN(LightningModule):
    def __init__(self, hparams=None):
        super(GCN, self).__init__()
        """
        Graph Convolutional Network for graph classification
        Parameters to be included in hparams
        ----------
        n_features : int
            Number of features for each node in the graph
            Default: 75 features for each atom in the molecule in the molecule dataset
        num_classes : int
            Number of classes for prediction
            Default : 2 (active or inactive)
        pool_type : str
            Type of pooling to aggregate the features after the graphconv layers
            Check : https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#module-torch_geometric.nn.glob
            Default : mean
            Options : mean, add, max
        activation : str
            One of 'relu', 'sigmoid' or 'tanh'.
            Default : 'relu'
            NOTE : prefinal layer has log_softmax
        opt : str
            One of 'adam' or 'adamax' or 'rmsprop'.
            Default : 'adam'
        batch_size: int
            Batch size for training.
            Default : 32
        lr: float
            Learning rate for optimizer.
            Default : 0.01
        weight_decay: float
            Weight decay in optimizer.
            Default : 0
        """
        self.__check_hparams(hparams)
        self.hparams = hparams

        # NOTE choose dataloaders appropriately
        self.dataloaders = MoleculeDataloaders()
        self.lenLoaders()
        self.telegrad_logs = {} # log everything you want to be reported via telegram here
        self.predicted_train = [] # to store the prediction in each epoch to calculate roc and prc
        self.true_train = [] # to store the true labels in each epoch to calculate roc and prc
        self.correct_train = 0 # number of correct predictions in each epoch
        self.predicted_val = [] # to store the prediction in each epoch to calculate roc and prc
        self.true_val = [] # to store the true labels in each epoch to calculate roc and prc
        self.correct_val = 0 # number of correct predictions in each epoch (validation)
        self.predicted_test = [] # to store the prediction in each epoch to calculate roc and prc
        self.true_test = [] # to store the true labels in each epoch to calculate roc and prc
        self.correct_test = 0 # number of correct predictions in each epoch (validation)

        self.conv1 = GCNConv(self.n_features, 128, cached=False) # if you defined cache=True, the shape of batch must be same!
        self.bn1 = BatchNorm1d(128)
        self.conv2 = GCNConv(128, 64, cached=False)
        self.bn2 = BatchNorm1d(64)
        self.fc1 = Linear(64, 64)
        self.bn3 = BatchNorm1d(64)
        self.fc2 = Linear(64, 64)
        self.fc3 = Linear(64, self.num_classes)

    def __check_hparams(self, hparams):
        self.n_features = hparams.n_features if hasattr(hparams,'n_features') else 75
        self.num_classes = hparams.num_classes if hasattr(hparams,'num_classes') else 2
        self.pool_type = hparams.pool_type if hasattr(hparams,'pool_type') else 'mean'
        self.opt = hparams.opt if hasattr(hparams,'opt') else 'adam'
        self.batch_size = hparams.batch_size if hasattr(hparams,'batch_size') else 32
        self.lr = hparams.lr if hasattr(hparams,'lr') else 0.01
        self.weight_decay = hparams.weight_decay if hasattr(hparams,'weight_decay') else 0
        self.activation = hparams.activation if hasattr(hparams,'activation') else 'relu'
        self.act = ACTS[self.activation]
        self.pool = POOL[self.pool_type]
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.act(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = self.act(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = self.pool(x, data.batch)
        x = self.act(self.fc1(x))
        x = self.bn3(x)
        x = self.act(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        # x = F.softmax(x, dim=1)
        return x 

    def configure_optimizers(self):
        """
            Choose Optimizer
        """
        return optimizers[self.opt](self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def getProbs(self,out):
        """
            Gives the probabilities of class-1 prediction
            (Need this because of log_softmax in the final layer)
        """
        num = np.exp(out)
        den = np.sum(num,1)
        den = np.stack((den,) * 2, axis=-1) 
        probs = num/den 
        return probs 

    def lenLoaders(self):
        self.train_dataloader_len = len(self.train_dataloader().dataset)
        self.test_dataloader_len = len(self.test_dataloader().dataset)
        self.val_dataloader_len = len(self.val_dataloader().dataset)

    def getAUC(self,y_pred,y_true):
        """
            Gives the ROC_AUC and PRC_AUC of the predicted class-1 probabilities
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        prc_auc = auc(recall,precision)
        roc_auc = roc_auc_score(y_true, y_pred)
        return roc_auc,prc_auc

    def update_arr(self,y,y_hat,true_arr,pred_arr,correct):
        pred = y_hat.max(dim=1)[1]
        correct += pred.eq(y).sum().item()
        true_arr.extend(np.array(y))
        pred_arr.extend(y_hat.detach().cpu().numpy())
        return correct


    def training_step(self, batch, batch_idx):
        x, y = batch, batch.y
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y) # negative log-likelihood loss
        logs = {'trainer_loss': loss}
        self.correct_train = self.update_arr(y,y_hat,self.true_train,self.predicted_train,self.correct_train)
        return {'loss': loss, 'log': logs}

    def training_epoch_end(self,outputs):
        avg_loss = torch.stack([x['trainer_loss'] for x in outputs]).mean()


        ops = np.array(self.predicted_train).reshape((int(self.batch_size*len(self.predicted_train)//self.batch_size),2))  
        probs = self.getProbs(ops)[:,1]
        acc = self.correct_train/self.train_dataloader_len
        self.correct_train = 0
        roc,prc = self.getAUC(probs,self.true_train)

        # telegrad logs
        self.telegrad_logs['lr'] = self.lr # for telegram bot
        logs = {'trainer_loss_epoch': avg_loss, 'roc_train':roc,'prc_train':prc,'acc_train':acc}
        self.logger.log_metrics({'learning_rate':self.lr}) # if lr is changed by telegram bot
        return {'train_loss': avg_loss, 'log':logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch, batch.y
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y) # negative log-likelihood loss
        self.correct_val = self.update_arr(y,y_hat,self.true_val,self.predicted_val,self.correct_val)
        return {'val_loss': loss}

    def validation_epoch_end(self,outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        ops = np.array(self.predicted_val).reshape((int(self.batch_size*len(self.predicted_val)//self.batch_size),2))  
        probs = self.getProbs(ops)[:,1]
        acc = self.correct_val/self.val_dataloader_len
        self.correct_val = 0
        roc,prc = self.getAUC(probs,self.true_val)

        # telegrad logs
        self.telegrad_logs={'roc':roc,'prc':prc,'acc':acc}
        logs = {'val_loss_epoch': avg_loss, 'roc_val':roc,'prc_val':prc,'acc_val':acc}
        self.logger.log_metrics({'learning_rate':self.lr}) # if lr is changed by telegram bot
        return {'val_loss': avg_loss, 'log':logs, 'prc_val':prc}


    def test_step(self, batch, batch_idx):
        x, y = batch, batch.y
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y) # negative log-likelihood loss
        self.correct_test = self.update_arr(y,y_hat,self.true_test,self.predicted_test,self.correct_test)
        return {'test_loss': loss}

    def test_epoch_end(self,outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()


        ops = np.array(self.predicted_test).reshape((int(self.batch_size*len(self.predicted_test)//self.batch_size),2))  
        probs = self.getProbs(ops)[:,1]
        acc = self.correct_test/self.test_dataloader_len
        self.correct_test = 0
        roc,prc = self.getAUC(probs,self.true_test)

        logs = {'test_loss_epoch': avg_loss, 'roc_test':roc,'prc_test':prc,'acc_test':acc}
        return {'test_loss': avg_loss,'roc_test':roc,'prc_test':prc,'acc_test':acc,'log':logs}





    def train_dataloader(self):
        return self.dataloaders.train_dataloader(self.batch_size)

    def val_dataloader(self):
        return self.dataloaders.val_dataloader(self.batch_size)
    
    def test_dataloader(self):
        return self.dataloaders.test_dataloader(self.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--n_features', type=int, default=75,
                            help='Number of features for each node in the graph')
        parser.add_argument('--num_classes', type=int, default=2,
                            help='Number of classes for prediction')
        parser.add_argument('--pool_type', type=str, default='mean',
                            help='Type of pooling to aggregate the features after the graphconv layers')
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
    parser = GCN.add_model_specific_args(parser)
    args = parser.parse_args()

    save_folder = 'model_weights/'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    early_stopping = EarlyStopping('val_loss')
    # saves checkpoints to 'save_folder' whenever 'val_loss' has a new min
    checkpoint_callback = ModelCheckpoint(
                            filepath=save_folder+'model_{epoch:02d}-{val_loss:.2f}')
    # tb_logger = loggers.TensorBoardLogger('logs')
    mlf_logger = MLFlowLogger(
                                experiment_name="gcn",
                                tracking_uri="file:./mlruns"
                                )

    # telegram
    token = telegram_config['token']
    user_id = telegram_config['user_id']
    bot = DLBot(token=token, user_id=user_id)
    telegramCallback = TelegramBotCallback(bot)

    model = GCN(args)

    trainer = Trainer(checkpoint_callback=checkpoint_callback,
                        early_stop_callback=early_stopping,
                        fast_dev_run=False,                     # make this as True only to check for bugs
                        max_epochs=1000,
                        min_epochs=20,
                        resume_from_checkpoint=None,            # change this to model_path
                        logger=mlf_logger,                      # mlflow logger
                        callbacks=[telegramCallback],           # telegrad
                        )

    trainer.fit(model)
    trainer.test()

if __name__ == "__main__":
    main()

        