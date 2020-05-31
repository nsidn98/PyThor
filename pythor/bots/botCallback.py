""" Deep Learning Telegram bot
DLBot and TelegramBotCallback classes for the monitoring and control
of a Keras \ Tensorflow training process using a Telegram bot
By: Eyal Zakkay, 2019
https://eyalzk.github.io/
"""

# from keras.callbacks import Callback
# import keras.backend as K

from pythor.bots.dl_bot import DLBot
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks import EarlyStopping
import signal
import sys

class TelegramBotCallback(Callback):
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
    """

    def __init__(self, kbot):
        assert isinstance(kbot, DLBot), 'Bot must be an instance of the DLBot class'
        super(TelegramBotCallback, self).__init__()
        self.kbot = kbot
        self.logs = {}
        # self.hist_logs = {} # to store values to plot

    def on_train_start(self, trainer, pl_module):
        self.logs['lr'] = pl_module.lr # Add learning rate to logs dictionary
        self.kbot.lr = self.logs['lr']  # Update bot's value of current LR
        self.kbot.activate_bot()  # Activate the telegram bot

        self.epochs = pl_module.current_epoch
        # loss history tracking
        self.loss_hist = []
        self.val_loss_hist = []
    
    def on_train_end(self, trainer, pl_module):
        self.kbot.send_message('Train Completed!')
        self.kbot.stop_bot()

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

        # LR handling
        # self.logs['lr'] = pl_module.lr
        self.kbot.lr = self.logs['lr']  # Update bot's value of current LR

        # make arrays for plotting 
        if pl_module.current_epoch == 0:
            self.kbot.hist_logs = pl_module.telegrad_logs.copy()
            for k in self.kbot.hist_logs.keys():
                self.kbot.hist_logs[k] = [self.kbot.hist_logs[k]] # make lists
        
        else:
            for k in self.kbot.hist_logs.keys():
                self.kbot.hist_logs[k].append(self.logs[k])
                
        # NOTE Deprecating original telegrad repo method because of the need to write a lot of boilerplate in lightning module
        # # Loss tracking
        # self.kbot.val_loss_hist.append(pl_module.val_loss)
        # self.logs['val_loss'] = pl_module.val_loss

        # self.kbot.loss_hist.append(pl_module.loss)
        # self.logs['loss'] = pl_module.loss

        # self.loss_hist.append(self.logs['loss'])
        # if 'val_loss' in logs:
        #     self.val_loss_hist.append(self.logs['val_loss'])
        # self.kbot.loss_hist = self.loss_hist
        # self.kbot.val_loss_hist = self.val_loss_hist

        # Epoch message handling
        tlogs = ', '.join([k+': '+'{:.4f}'.format(v) for k, v in zip(self.logs.keys(), self.logs.values())])  # Clean logs string
        message = 'Epoch %d/%d \n' % (pl_module.current_epoch + 1, trainer.max_epochs) + tlogs
        # Send epoch end logs
        if self.kbot.verbose:
            self.kbot.send_message(message)
        # Update status message
        self.kbot.set_status(message)

