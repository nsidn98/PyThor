"""
    Advantage Actor-Critic for vector-based states
    Uses multiple workers to avoid the use of a replay buffer.
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
from torch.distributions import Normal

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from pythor.RL.Value.utils.replay_buffer import ReplayBuffer
from pythor.datamodules.rl_dataloader import RLDataset, Experience 
from pythor.RL.common.multiprocessing_env import SubprocVecEnv
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


class AC_Linear(nn.Module):
    """
        Actor and Critic for A2C
        Parameters:
        -----------
        input_shape : int
            The size of the state vector
        num_actions : int
            Number of actions available for a time-step
        hidden_size : list of ints
            List of hidden layers dimensions
            NOTE: the final layer single neuron will be added automatically
        activation : str
            Non linear activation for layers
            Default : 'relu'
            Options : 'relu', 'tanh', 'sigmoid'   
        act_type : str
            Type of action for the environment
            Default : 'discrete'
            Choices : 'discrete'(for discrete actions) and 'cont' for continuous actions
        std : float
            init for standard deviation for continuous actions
    """
    def __init__(self, input_shape, num_actions, hidden_actor, hidden_critic, activation, act_type, std=0.0):
        super(AC_Linear, self).__init__()
        self.act = ACTS[activation]
        self.act_type = act_type

        linear_critic = [nn.Linear(in_f, out_f) for in_f, out_f in 
                    zip([input_shape + num_actions]+hidden_critic, hidden_critic+[1])] # +[1] is the output layer neuron getting added

        self.critic = nn.Sequential(
            *[layer for tup in zip(linear_critic, [self.act() for _ in linears]) for layer in tup][:-1]
        )
        
        linear_actor = [nn.Linear(in_f, out_f) for in_f, out_f in 
                    zip([input_shape + num_actions]+hidden_actor, hidden_actor+[num_actions])] # +[1] is the output layer neuron getting added

        self.actor = nn.Sequential(
            *[layer for tup in zip(linear_actor, [self.act() for _ in linears]) for layer in tup][:-1]
        )

        # continuous actions
        if self.act_type == 'cont':
            self.log_std = nn.Parameter(torch.ones(1, num_actions) * std)
        
        self.apply(self.init_weights)
        
    def forward(self, x):
        """
            Will output a value for the current state
            and a distribution to sample action from
        """
        value = self.critic(x.float())
        mu = self.actor(x)
        if self.act_type == 'discrete':
            probs = torch.softmax(mu,dim=1)
            dist = Categorical(probs)
        if self.act_type == 'cont':
            std   = self.log_std.exp().expand_as(mu)
            dist  = Normal(mu, std)
        return dist, value

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0., std=0.1)
            nn.init.constant_(m.bias, 0.1)
    
class AC_CNN(nn.Module):
    """
        Actor and Critic for A2C for image based states
        Parameters:
        -----------
        input_shape : int
            The size of the state vector
        num_actions : int
            Number of actions available for a time-step
        hidden_size : list of ints
            List of hidden layers dimensions
            NOTE: the final layer single neuron will be added automatically
        activation : str
            Non linear activation for layers
            Default : 'relu'
            Options : 'relu', 'tanh', 'sigmoid'   
        act_type : str
            Type of action for the environment
            Default : 'discrete'
            Choices : 'discrete'(for discrete actions) and 'cont' for continuous actions
        std : float
            init for standard deviation for continuous actions
    """
    def __init__(self, input_shape, num_actions, hidden_actor, hidden_critic, activation, act_type, std=0.0):
        super(AC_CNN, self).__init__()
        self.act = ACTS[activation]
        self.act_type = act_type

        linear_critic = [nn.Linear(in_f, out_f) for in_f, out_f in 
                    zip([input_shape + num_actions]+hidden_critic, hidden_critic+[1])] # +[1] is the output layer neuron getting added

        self.critic = nn.Sequential(
            *[layer for tup in zip(linear_critic, [self.act() for _ in linears]) for layer in tup][:-1]
        )
        
        linear_actor = [nn.Linear(in_f, out_f) for in_f, out_f in 
                    zip([input_shape + num_actions]+hidden_actor, hidden_actor+[num_actions])] # +[1] is the output layer neuron getting added

        self.actor = nn.Sequential(
            *[layer for tup in zip(linear_actor, [self.act() for _ in linears]) for layer in tup][:-1]
        )

        # continuous actions
        if self.act_type == 'cont':
            self.log_std = nn.Parameter(torch.ones(1, num_actions) * std)
        
        self.apply(self.init_weights)
        
    def forward(self, x):
        """
            Will output a value for the current state
            and a distribution to sample action from
        """
        value = self.critic(x.float())
        mu = self.actor(x)
        if self.act_type == 'discrete':
            probs = torch.softmax(mu,dim=1)
            dist = Categorical(probs)
        if self.act_type == 'cont':
            std   = self.log_std.exp().expand_as(mu)
            dist  = Normal(mu, std)
        return dist, value

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0., std=0.1)
            nn.init.constant_(m.bias, 0.1)


class A2C(LightningModule):
    """ A2C main module """

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
        self.value_lr = hparams.value_lr # for telegrad
        self.policy_lr = hparams.policy_lr
        self.lr = self.value_lr # just to bypass rl_bot (Callback)
        self.reward_hist = []
        

        # if self.env_type == 'linear':
        self.env = self.make_env()
        self.test_env = self.make_env()
        obs_shape = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.shape[0]
        self.value_net        = ValueNetwork(obs_shape, n_actions,[256], hparams.activation, hparams.init_w)
        self.target_value_net = ValueNetwork(obs_shape, n_actions,[256], hparams.activation, hparams.init_w)
        self.policy_net = PolicyNetwork(obs_shape, n_actions, [256], hparams.activation, hparams.init_w)
        self.target_policy_net = PolicyNetwork(obs_shape, n_actions, [256], hparams.activation, hparams.init_w)
                
        self.total_reward = 0
        self.episode_reward = 0

        self.state = self.env.reset()
        


    def make_envs(self):
        """
            Make parallel envs
        """
        num_envs = self.hparams.num_envs
        def make_env():
            def _thunk():
                if self.env_type == 'linear':
                    env = gym.make(self.env_name)
                if self.env_type == 'cnn':
                    env = make_atari(self.env_name)
                    env = wrap_deepmind(env)
                    env = wrap_pytorch(env)
                # env = NormalizedActions(env) # normalize actions
                return env
            return _thunk

        envs = [make_env() for i in range(num_envs)]
        envs = SubprocVecEnv(envs)
        
        return envs

    def test_environment(self):
        """
            Test the environment after self.hparams.test_env_steps 
        """
        state = self.test_env.reset()
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist, _ = self.model(state)
            next_state, reward, done, _ = self.test_env.step(dist.sample().cpu().numpy()[0])
            state = next_state
            total_reward += reward
        return total_reward


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Passes in a state x through the network and gets the action as an output
            Args:
                x: environment state
            Returns:
                actions
        """
        output = self.policy_net(x.float())
        return output

    def compute_returns(self, next_value, rewards, masks):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.hparams.gamma * R * masks[step]
            returns.insert(0, R)
        return returns


    def run_steps(self):
        """
            Run a few steps(self.hparams.num_steps) in the parallel environments
            and store the values, rewards, log_probs, masks and entropies
        """
        log_probs = []
        values    = []
        rewards   = []
        masks     = []
        entropy = 0
        for _ in range(self.hparams.num_steps):
            state = torch.FloatTensor(self.state).to(device)
            dist, value = self.model(state)

            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
            
            self.state = next_state
            frame_idx += 1
            
            if self.global_step % 1000 == 0:
                test_rewards.append(np.mean([test_env() for _ in range(10)]))
                # plot(frame_idx, test_rewards)

        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        returns = compute_returns(next_value, rewards, masks)
        
        log_probs = torch.cat(log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(values)

        advantage = returns - values

        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

    return loss

    def run_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        device = self.get_device(batch)
        epsilon = self.get_epsilon()
        
        reward, done = self.agent.play_step(self.policy_net, epsilon, self.global_step, device)
        self.episode_reward += reward

        self.log = {'total_reward': torch.tensor(self.total_reward).to(device),
               'steps': torch.tensor(self.global_step).to(device)}

        self.progress_bar = self.log.copy()

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0
            self.log['episode_reward'] = torch.tensor(self.total_reward).to(device)
            self.telegrad_rewards.append(self.total_reward) # telegrad
            self.reward_hist.append(self.total_reward)
        
        if self.global_step % self.hparams.test_env_steps == 0:
            test_reward = self.test_environment()
            self.log['test_reward'] = torch.tensor(test_reward).to(device)
            self.telegrad_test_rewards.append(test_reward)

    def training_step(self, batch, batch_idx):
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number
        Returns:
            Training loss and log metrics
        """
        if optimizer_idx == 0:
            self.run_step(batch)
            self.ddpg_policy_loss(batch)
            loss = self.policy_loss
        if optimizer_idx == 1:
            self.ddpg_value_loss(batch)
            loss = self.value_loss

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        return OrderedDict({'loss': loss, 'log': self.log, 'progress_bar': self.progress_bar})

    def training_epoch_end(self, outputs):
        log = {'learning_rate': self.lr}
        # NOTE the stop flag is only with training rewards and not with test rewards
        stop_flag, mean_rew = self.check_convergence()
        self.telegrad_logs={'rewards':self.telegrad_rewards.copy(), 
                            'lr':self.value_lr, 
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
        optimizer = optimizers[self.hparams.opt](self.model.parameters(), lr=self.lr, weight_decay=self.hparams.weight_decay)
        
        return optimizer

    def check_convergence(self):
        stop_training = False
        mean_rew = np.mean(np.array(self.reward_hist))
        if len(self.reward_hist) > 100:
            mean_rew = np.mean(np.array(self.reward_hist[:-100]))
            if mean_rew > self.hparams.env_max_rew:
                stop_training = True
        return stop_training, mean_rew

    def train_dataloader(self):
        pass
    # def train_dataloader(self) -> DataLoader:
    #     """Initialize the Replay Buffer dataset used for retrieving experiences"""
    #     dataset = RLDataset(self.agent.replay_buffer, self.hparams.episode_length)

    #     dataloader = DataLoader(dataset=dataset,
    #                             batch_size=self.hparams.batch_size,
    #                             )
    #     return dataloader

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
                    accumulate_grad_batches=1
    )
    model = DDPG(hparams)
    trainer.fit(model)


if __name__ == '__main__':
    # torch.manual_seed(0)
    # np.random.seed(0)
    parser = argparse.ArgumentParser()
    # algo
    parser.add_argument("--algo_name", type=str, default='ddpg',
                        help='Name of RL algorithm')
    parser.add_argument("--soft_tau", type=float, default=1e-2, help="weight for soft update of networks")
    parser.add_argument("--init_w", type=float, default=1e-3, help="weight initialization limits for networks")
    # parser.add_argument("--noisy", type=int, default=0, help="Whether to use noisy networks or not")
    # environment
    parser.add_argument("--env_name", type=str, default="Pendulum-v0", help="gym environment tag")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument('--env_type', type=str, default='linear', choices=['linear', 'cnn'], help= 'type of network to use')
    parser.add_argument("--episode_length", type=int, default=1000, help="max length of an episode")
    parser.add_argument("--env_max_rew", type=int, default=195, help="avg rewards in 100 episodes to stop training")
    parser.add_argument("--num_steps", type=int, default=5, help="run the environment for these many steps to collect rollouts")
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
    parser.add_argument("--eps_last_frame", type=int, default=50000, help="what frame should epsilon stop decaying")
    parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
    parser.add_argument("--eps_end", type=float, default=0.01, help="final value of epsilon")
    # neural network
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate for value network")
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid', 'tanh'], help='activations for nn layers')
    # optimizer
    parser.add_argument('--opt', type=str, default='adam', choices=['adam', 'adamax', 'rmsprop'], help='optimizer type for optimization')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay in optimizer')
    args = parser.parse_args()

    main(args)
