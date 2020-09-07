"""
    Proximal Policy Optimization for vector-based states
    # TODO add networks for image-based states
"""
import os
import random
import gym
import numpy as np
import collections
from typing import Tuple, List
import argparse
from collections import OrderedDict
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
# torch.autograd.set_detect_anomaly(True)

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from pythor.RL.Value.utils.replay_buffer import ReplayBuffer
from pythor.datamodules.rl_dataloader import RLDataset, Experience 
from pythor.RL.common.OU_noise import OUNoise # to add Ornstein Uhlenbeck Noise (DDPG special)
from pythor.RL.common.normalise_actions import NormalizedActions # to normalise actions
# telegrad
from pythor.bots.rlCallback import TelegramRLCallback
from pythor.bots.rl_bot import RLBot
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
