""" Deep Learning Telegram bot
Bot for monitoring RL agent training
Original Code By: Eyal Zakkay, 2019
https://eyalzk.github.io/
"""
import mlflow
import mlflow.pytorch
from pythor.bots.rl_bot import RLBot
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks import EarlyStopping
import signal
import numpy as np
import sys

class TelegramRLCallback(Callback):
    """Callback that sends metrics and responds to Telegram Bot.
    Supports the following commands:
     /start: activate automatic updates every epoch
     /help: get a reply with all command options
     /status: get a reply with the latest epoch's results
     /getlr: get a reply with the current learning rate
     /setlr: change the learning rate (multiply by a factor of 0.5,0.1,2 or 10)
     /plot: get a reply with the loss convergence plot image
     /quiet: stop getting automatic updates each epoch
     /stoptraining: kill Keras training process
    # Arguments
        kbot: Instance of the DLBot class, holding the appropriate bot token
    # Raises
        TypeError: In case kbot is not a DLBot instance.
    Usage: store rl rewards in self.telegrad_logs={'reward':[0.001, 0.1,..., 0.2]} in training_step
    """

    def __init__(self, kbot):
        assert isinstance(kbot, RLBot), 'Bot must be an instance of the DLBot class'
        super(TelegramRLCallback, self).__init__()
        self.kbot = kbot
        self.logs = {}
        # self.hist_logs = {} # to store values to plot

    def on_train_start(self, trainer, pl_module):
        self.logs['lr'] = pl_module.lr # Add learning rate to logs dictionary
        self.kbot.lr = self.logs['lr']  # Update bot's value of current LR
        self.kbot.activate_bot()  # Activate the telegram bot NOTE check here for internet connections

        self.epochs = pl_module.current_epoch
    
    def on_train_end(self, trainer, pl_module):
        self.kbot.send_message('Train Completed!')
        self.kbot.stop_bot()
        # trainer.checkpoint_callback.on_epoch_end()
        mlflow.pytorch.log_model(pl_module.net, "runs:/" + pl_module.logger.run_id + "/model_weights")

    def on_epoch_start(self, trainer, pl_module):
        if self.kbot.modify_lr != 1:
            current_lr = float(pl_module.lr) # get current lr
            # new LR
            new_lr = current_lr*self.kbot.modify_lr
            pl_module.lr = new_lr
            self.kbot.modify_lr = 1 # Set multiplier back to 1
            message = '\nEpoch %05d: setting learning rate to %s.' % (pl_module.current_epoch + 1, pl_module.lr)
            print(message)
            self.kbot.send_message(message)

    def on_epoch_end(self, trainer, pl_module):
        """
            Store all your logs you want to plot in telegrad in self.telegrad_logs dict
            Give only telegrad_logs={'rewards':list, 'lr': self.lr}
        """

        # Did user invoke STOP command 
        # NOTE Change HERE
        if self.kbot.stop_train_flag:
            # NOTE Change HERE
            # model.stop_training =True
            self.kbot.send_message('Training Stopped!')
            print('Training Stopped! Stop command sent via Telegram bot.')
            raise KeyboardInterrupt

        self.logs = pl_module.telegrad_logs.copy()
        # print(self.logs)
        # LR handling
        # self.logs['lr'] = pl_module.lr
        self.kbot.lr = self.logs['lr']  # Update bot's value of current LR

        # make arrays for plotting 
        if pl_module.current_epoch == 0:
            self.kbot.hist_logs = pl_module.telegrad_logs.copy()
            for k in self.kbot.hist_logs.keys():
                if k != 'rewards':
                    self.kbot.hist_logs[k] = [self.kbot.hist_logs[k]] # make lists
                else:
                    self.kbot.hist_logs[k] = self.kbot.hist_logs[k]
        
        else:
            for k in self.kbot.hist_logs.keys():
                # self.kbot.hist_logs[k].append(self.logs[k])
                if type(self.logs[k])==list:
                    self.kbot.hist_logs[k].extend(self.logs[k]) # changing to extend because telegrad_logs only has rewards stored in a list
                else:
                    self.kbot.hist_logs[k].append(self.logs[k])
                

        # Epoch message handling
        tlogs = 'Episodes: ' + str(len(self.kbot.hist_logs['rewards'])) + '\nLatest Rewards Avg: ' + str(self.logs['mean_rew']) +\
                '\nSteps: ' + str(pl_module.global_step)
        message = 'Epoch %d/%d \n' % (pl_module.current_epoch + 1, trainer.max_epochs) + tlogs
        # tlogs = ', '.join([k+': '+'{:.4f}'.format(v) for k, v in zip(self.logs.keys(), self.logs.values())])  # Clean logs string
        # message = 'Epoch %d/%d \n' % (pl_module.current_epoch + 1, trainer.max_epochs) + tlogs
        # NOTE removing epoch updates
        # Send epoch end logs
        # if self.kbot.verbose:
        #     self.kbot.send_message(message)
        ###################################
        # Update status message
        self.kbot.set_status(message)

