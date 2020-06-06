import os
import importlib
import math, random
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
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.dataset import IterableDataset

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from pythor.RL.Value.utils.replay_buffer import ReplayBuffer, PrioritizedBuffer
from pythor.datamodules.rl_dataloader import RLDataset, PriorityRLDataset, Experience
# from pythor.RL.Value.dqn import DQNAgent, DQNetwork, CnnDQNetwork
from pythor.RL.common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
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

class ValueRL(LightningModule):
    """ Value based RL algorithms """

    def __init__(self, hparams: argparse.Namespace) -> None:
        super().__init__()
        self.hparams = hparams
        self.gamma = hparams.gamma
        self.env_name = hparams.env_name # example: 'CartPole-v0
        self.env_type = hparams.env_type # 'linear' or 'cnn'

        # telegrad
        self.telegrad_logs = {}
        self.telegrad_rewards = []
        self.telegrad_test_rewards = []
        self.lr = hparams.lr # for telegrad
        self.reward_hist = []

        # import algo file
        import_string = 'pythor.RL.Value.'
        import_string += self.hparams.algo_name # EXAMPLE: 'pythor.RL.Value.dqn' to import classes from pythor.RL.Value.dqn
        algo = importlib.import_module(import_string) # EXAMPLE: this is substitute for import pythor.RL.Value.dqn

        if self.env_type == 'linear':
            self.env = self.make_env()
            self.test_env = self.make_env()
            obs_shape = self.env.observation_space.shape[0]
            n_actions = self.env.action_space.n
            self.net = algo.Network(obs_shape, n_actions, hparams.activation, hparams)
            self.target_net = None
            # self.net = DQNetwork(obs_shape, n_actions, hparams.activation)
            if self.hparams.algo_name != 'dqn':
                self.target_net = algo.Network(obs_shape, n_actions, hparams.activation, hparams)

        if self.env_type == 'cnn':
            self.env = self.make_env()
            self.test_env = self.make_env()
            obs_shape = self.env.observation_space.shape # example (480,280,3)
            obs_shape = np.roll(np.array(obs_shape),1) # (3, 480, 280)
            n_actions = self.env.action_space.n
            self.net = algo.CnnNetwork(obs_shape, n_actions, hparams.activation, hparams)
            self.target_net = None
            # self.net = CnnDQNetwork(obs_shape, n_actions, hparams.activation)
            if self.hparams.algo_name != 'dqn':
                self.target_net = algo.CnnNetwork(obs_shape, n_actions, hparams.activation, hparams)
        
        if self.hparams.priority:
            self.buffer = PrioritizedBuffer(self.hparams.replay_size, beta_start=self.hparams.beta_start, beta_frames=self.hparams.beta_frames)
        else:
            self.buffer = ReplayBuffer(self.hparams.replay_size)

        self.agent = algo.Agent(self.env, self.buffer, self.hparams)

        self.agent.update_target(self.net,self.target_net) # sync the networks
        
        self.total_reward = 0
        self.episode_reward = 0
        
        self.populate(self.hparams.warm_start_steps)

    def make_env(self):
        if self.env_type == 'linear':
            env = gym.make(self.env_name)
        if self.env_type == 'cnn':
            env = make_atari(self.env_name)
            env = wrap_deepmind(env)
            env = wrap_pytorch(env)
        return env


    def populate(self, steps: int = 1000) -> None:
        """
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences
        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes in a state x through the network and gets the q_values of each action as an output
        Args:
            x: environment state
        Returns:
            q values
        """
        # output = self.net(x)
        output = self.net(x.float())
        return output

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0].device.index if self.on_gpu else 'cpu'

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number
        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        epsilon = max(self.hparams.eps_end, self.hparams.eps_start -
                      self.global_step + 1 / self.hparams.eps_last_frame)
        
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward

        if self.hparams.noisy:
            self.net.reset_noise()
            if self.target_net:
                self.target_net.reset_noise()

        # calculate training loss
        if self.hparams.priority:
            # loss and updates for priority buffer
            loss, indices, priors = self.agent.priority_compute_td_loss(self.net, self.target_net, batch)
            self.agent.replay_buffer.update_priorities(indices,priors.data.cpu().numpy())
            self.agent.replay_buffer.update_beta(self.global_step)
        else:
            loss = self.agent.compute_td_loss(self.net, self.target_net, batch)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        log = {'total_reward': torch.tensor(self.total_reward).to(device),
               'reward': torch.tensor(reward).to(device),
               'steps': torch.tensor(self.global_step).to(device)}

        progress_bar = log.copy()

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0
            log['episode_reward'] = torch.tensor(self.total_reward).to(device)
            self.telegrad_rewards.append(self.total_reward) # telegrad
            self.reward_hist.append(self.total_reward)
        
        if self.global_step % self.hparams.test_env_steps == 0:
            test_reward = self.test_environment()
            log['test_reward'] = test_reward
            self.telegrad_test_rewards.append(test_reward)

        # Soft update of target network NOTE Check according to algo
        if self.global_step % self.hparams.sync_rate == 0:
            self.agent.update_target(self.net,self.target_net)

        # self.telegrad_rewards.append(self.total_reward) # telegrad
        return OrderedDict({'loss': loss, 'log': log, 'progress_bar': progress_bar})

    def training_epoch_end(self, outputs):
        log = {'learning_rate': self.lr}
        # NOTE the stop flag is only with training rewards and not with test rewards
        stop_flag, mean_rew = self.check_convergence()
        self.telegrad_logs={'rewards':self.telegrad_rewards.copy(), 
                            'lr':self.lr, 
                            'mean_rew':np.round(mean_rew,3),
                            'test_rewards':self.telegrad_test_rewards.copy()} # telegrad
        self.telegrad_rewards.clear() # clear the list
        self.telegrad_test_rewards.clear()
        if stop_flag:
            print('#'*50)
            print('\nStopping because rewards converged\n')
            print('#'*50)
            raise KeyboardInterrupt
        return {'log':log}

    def configure_optimizers(self):
        """ Initialize Adam optimizer"""
        optimizer = optimizers[self.hparams.opt](self.net.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return [optimizer]

    def check_convergence(self):
        stop_training = False
        mean_rew = np.mean(np.array(self.reward_hist))
        if len(self.reward_hist) > 100:
            mean_rew = np.mean(np.array(self.reward_hist[:-100]))
            if mean_rew > self.hparams.env_max_rew:
                stop_training = True
        return stop_training, mean_rew

    def test_environment(self):
        """
            Test the environment after self.hparams.test_env_steps 
        """
        state = self.test_env.reset() 
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.net(state)
            _, action = torch.max(q_values, dim=1)
            # NOTE that action has been converted to int. May need to change to float for custom envs
            action = int(action.item())
            next_state, reward, done, _ = self.test_env.step(action)
            state = next_state
            total_reward += reward
        return total_reward

    def train_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        if self.hparams.priority:
            dataset = PriorityRLDataset(self.agent.replay_buffer, self.hparams.episode_length)
        else:
            dataset = RLDataset(self.agent.replay_buffer, self.hparams.episode_length)

        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.hparams.batch_size,
                                )
        return dataloader
        
def main(hparams) -> None:
    experiment_name = hparams.algo_name

    save_folder = 'model_weights/' + experiment_name
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    
    checkpoint_callback = ModelCheckpoint(
                            filepath=save_folder+'/model_{epoch:02d}')

    mlf_logger = MLFlowLogger(
                                experiment_name=experiment_name,
                                tracking_uri="file:./mlruns"
                                )

    # telegram
    token = telegram_config['token']
    user_id = telegram_config['user_id']
    bot = RLBot(token=token, user_id=user_id)
    telegramCallback = TelegramRLCallback(bot)
    trainer = Trainer(checkpoint_callback=checkpoint_callback,
        max_epochs=10000,
        early_stop_callback=False,
        val_check_interval=100,
        logger=mlf_logger,
        callbacks=[telegramCallback],
    )
    model = ValueRL(hparams)
    trainer.fit(model)


if __name__ == '__main__':
    # torch.manual_seed(0)
    # np.random.seed(0)
    parser = argparse.ArgumentParser()
    # algo
    parser.add_argument("--algo_name", type=str, default='dqn', choices=['dqn','ddqn', 'dddqn'],
                        help='Name of RL algorithm to use')
    parser.add_argument("--noisy", type=int, default=0, help="Whether to use noisy networks or not")
    # environment
    parser.add_argument("--env_name", type=str, default="CartPole-v0", help="gym environment tag")
    parser.add_argument('--env_type', type=str, default='linear', choices=['linear', 'cnn'], help= 'type of network to use')
    parser.add_argument("--episode_length", type=int, default=200, help="max length of an episode")
    parser.add_argument("--env_max_rew", type=int, default=195, help="avg rewards in 100 episodes to stop training")
    parser.add_argument("--warm_start_steps", type=int, default=1000, help="warm up steps in the environment before training")
    parser.add_argument("--test_env_steps", type=int, default=1000, help="Test environment after these man steps")
    # rl agent
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--sync_rate", type=int, default=10, help="how many frames do we update the target network")
    # replay buffer
    parser.add_argument("--replay_size", type=int, default=100000, help="capacity of the replay buffer")
    parser.add_argument("--priority", type=int, default=0, help="Whether use priority buffer or not")
    parser.add_argument("-beta_start", type= float, default=0.4, help="Initial beta for priority buffer")
    parser.add_argument("-beta_frames", type= int, default=1000, help="Number of frames for increase beta to 1.0 for priority buffer")
    # exploration
    parser.add_argument("--eps_last_frame", type=int, default=1000, help="what frame should epsilon stop decaying")
    parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
    parser.add_argument("--eps_end", type=float, default=0.01, help="final value of epsilon")
    # neural network
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid', 'tanh'], help='activations for nn layers')
    # optimizer
    parser.add_argument('--opt', type=str, default='adam', choices=['adam', 'adamax', 'rmsprop'], help='optimizer type for optimization')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay in optimizer')
    args = parser.parse_args()

    main(args)
