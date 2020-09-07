"""
    DQN with Hindsight Experience Replay(HER)
"""
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
import torch.nn.functional as F
from torch.autograd import Variable

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from pythor.RL.Value.utils.replay_buffer import ReplayBuffer
from pythor.datamodules.rl_dataloader import RLDataset, Experience
from pythor.RL.common.layers import NoisyLinear # for noisy networks
# from PyThor.pythor.RL.common.wrappers import make_atari, wrap_deepmind, wrap_pytorch

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

class Network(nn.Module):
    """
        Network
        Parameters:
        -----------
        input_shape : int
            State vector dimension
        num_actions : int
            Number of actions
        activation : str
            Non linear activation for layers
            Default : 'relu'
            Options : 'relu', 'tanh', 'sigmoid'
        hparams.noisy : bool
            Whether to use noisy networks (https://arxiv.org/abs/1706.10295) for exploration
            Default : False
        Can be used for OpenAI gym type environments like CartPole-v0, etc.
    """
    def __init__(self, input_shape, num_actions,activation, hparams):
        super(Network, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.act = ACTS[activation]
        self.noisy = hparams.noisy

        linears = [nn.Linear(in_f, out_f) for in_f, out_f in 
                zip([input_shape]+hidden_size, hidden_size+[num_actions])] # +[num_actions] is the output layer neurons getting added
        self.fc = nn.Sequential(
        *[layer for tup in zip(linears, [self.act() for _ in linears]) for layer in tup][:-1]
        )

    def forward(self, state, goal):
        x = torch.cat([state, goal],dim=1)
        return self.layers(x.float())



class Agent:
    """
    Base Agent class handling the interaction with the environment

    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """
    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer, hparams: argparse.Namespace):
        self.hparams = hparams
        self.replay_buffer = replay_buffer
        self.env = env

        self.reset()
        self.state = self.env.reset()

    def reset(self):
        """ Resets the environment and updates the state"""
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str):
        """
        Using the given network, decide what action to carry out
        using an epsilon-greedy policy

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor([self.state])

            if device not in ['cpu']:
                state = state.cuda(device)

            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            # NOTE that action has been converted to int. May need to change to float for custom envs
            action = int(action.item())

        return action
    
    def compute_td_loss(self, net: nn.Module, target_net: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]):
        states, actions, rewards, dones, next_states = batch
        dones = dones.type(torch.FloatTensor)
        q_values = net(states)
        with torch.no_grad():
            next_q_values = net(next_states)

        q_value          = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = rewards + self.hparams.gamma * next_q_value * (1 - dones)
        
        # loss = (q_value - (expected_q_value.data)).pow(2).mean()
        loss = nn.MSELoss()(q_value,expected_q_value)

        return loss
    
    def priority_compute_td_loss(self, net: nn.Module, target_net: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]):
        """
            If priority buffer used
        """
        states, actions, rewards, dones, next_states, indices, weights = batch

        dones = dones.type(torch.FloatTensor)
        q_values = net(states)
        with torch.no_grad():
            next_q_values = net(next_states)

        q_value          = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = rewards + self.hparams.gamma * next_q_value * (1 - dones)
        
        loss = (q_value - (expected_q_value.data)).pow(2)*weights
        priors = loss + 1e-5
        loss = loss.mean()
        # loss = nn.MSELoss()(q_value,expected_q_value)

        return loss, indices, priors
    

    @torch.no_grad()
    def play_step(self, net: nn.Module, epsilon: float = 0.0, device: str = 'cpu'):
        """
        Carries out a single interaction step between the agent and the environment

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """

        action = self.get_action(net,epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)
        # done = float(done)

        exp = Experience(self.state, action, reward, done, new_state)

        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done

    def update_target(self, net:nn.Module, target_net:nn.Module):
        # target_net.load_state_dict(net.state_dict())
        # Not required for DQN
        pass
