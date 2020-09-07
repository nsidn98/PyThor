"""
    New algo
"""
import os
import random
import gym
import numpy as np
import collections

from torch.utils.data import IterableDataset

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


# Named tuple for storing experience steps gathered in training
Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])

class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        """
        Add experience to the buffer
        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool), np.array(next_states))

class NormalizedActions(gym.ActionWrapper):
    """
        Normalise the actions
    """

    def action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)
        
        return action

    def reverse_action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        
        return actions

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the ReplayBuffer
    which will be updated with new experiences during training
    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]

class ValueNetwork(nn.Module):
    """
        Value Network
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
        init_w : float
            Weight initialization for the final layer (weights and bias) 
            (Uniform initialization)
        NOTE: Make sure that the state and the action vectors are normalised to 
        the same range
        NOTE: In image based states, have to concatanate the output obtained 
        after the conv layers with the actions. Take care with the normalisation
    """
    def __init__(self, input_shape, num_actions, hidden_size, activation, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        self.act = ACTS[activation]

        linears = [nn.Linear(in_f, out_f) for in_f, out_f in 
                    zip([input_shape + num_actions]+hidden_size, hidden_size+[1])] # +[1] is the output layer neuron getting added
        self.fc = nn.Sequential(
            *[layer for tup in zip(linears, [self.act() for _ in linears]) for layer in tup][:-1]
        )

        # initialize the weights and bias of the last layer
        self.fc[-1].weight.data.uniform_(-init_w, init_w)
        self.fc[-1].bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state.float(), action.float()],1)
        x = self.fc(x)
        return x
    
class PolicyNetwork(nn.Module):
    """
        Value Network
        Parameters:
        -----------
        input_shape : int
            The size of the state vector
        num_actions : int
            Number of actions available for a time-step
        hidden_size : list of ints
            List of hidden layers dimensions
            NOTE: the final layer output dimensions will be added automatically
        activation : str
            Non linear activation for layers (except final layer as it has tanh)
            Default : 'relu'
            Options : 'relu', 'tanh', 'sigmoid'   
        init_w : float
            Weight initialization for the final layer (weights and bias) 
            (Uniform initialization)
    """
    def __init__(self, input_shape, num_actions, hidden_size, activation, init_w=3e-3):
        super(PolicyNetwork, self).__init__()
        self.act = ACTS[activation]

        linears = [nn.Linear(in_f, out_f) for in_f, out_f in 
                    zip([input_shape]+hidden_size, hidden_size+[num_actions])] # +[num_actions] is the output layer neurons getting added

        self.fc = nn.Sequential(
            *[layer for tup in zip(linears, [self.act() for _ in linears]) for layer in tup][:-1]
        )

        # initialize the weights and bias of the last layer
        self.fc[-1].weight.data.uniform_(-init_w, init_w)
        self.fc[-1].bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = self.fc(state.float())
        x = torch.tanh(x)
        return x
    
    def get_action(self, state):
        state  = torch.FloatTensor(state).unsqueeze(0)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0, 0]

class Agent:
    """
        Base Agent class handling the interaction with the environment

        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
            hparams: params from lightning module
    """
    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer, hparams: argparse.Namespace):
        self.hparams = hparams
        self.replay_buffer = replay_buffer
        self.env = env
        self.ou_noise = OUNoise(self.env.action_space)  # NOTE: for custom envs it is necessary to have action_space defined
        self.episode_steps = 0

        self.reset()
        self.state = self.env.reset()

    def reset(self):
        """ Resets the environment and updates the state"""
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, step: int, device: str):
        """
        Using the given network, decide what action to carry out
        using an epsilon-greedy policy

        Args:
            net: policy network
            epsilon: value to determine likelihood of taking a random action
            step: current time_step 
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

            action = net(state)
            action = action.detach().cpu().numpy()[0, 0]

        action = self.ou_noise.get_action(action, step) # add OU noise

        return action

    @torch.no_grad()
    def play_step(self, net: nn.Module, epsilon: float = 0.0, step: int = 0, device: str = 'cpu'):
        """
        Carries out a single interaction step between the agent and the environment

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            step: current time_step
            device: current device

        Returns:
            reward, done
        """
        self.episode_steps += 1
        action = self.get_action(net, epsilon, step, device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)
        # done = float(done)

        exp = Experience(self.state, action, reward, done, new_state)

        self.replay_buffer.append(exp)

        self.state = new_state

        # terminate the episode after hparams.episode_length
        if self.episode_steps == self.hparams.episode_length:
            done = 1

        if done:
            self.episode_steps = 0
            self.reset()
        return reward, done

    def soft_update_networks(self, net, target_net, soft_tau):
        """
            Soft update of networks
        """
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
    


class myModule(LightningModule):
    """ main module """

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
        
        self.buffer = ReplayBuffer(self.hparams.replay_size)

        self.agent = Agent(self.env, self.buffer, self.hparams)

        self.agent.soft_update_networks(self.value_net,self.target_value_net, soft_tau=1) # sync the networks
        self.agent.soft_update_networks(self.policy_net,self.target_policy_net, soft_tau=1) # sync the networks
        
        self.total_reward = 0
        self.episode_reward = 0
        
        self.populate(self.hparams.warm_start_steps)

    def loss(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        state, action, reward, done, next_state = batch
        done = done.type(torch.FloatTensor)

        # update networks first
        self.agent.soft_update_networks(self.policy_net, self.target_policy_net, self.hparams.soft_tau)
        self.agent.soft_update_networks(self.value_net, self.target_value_net, self.hparams.soft_tau)

        policy_loss = self.value_net(state, self.policy_net(state))
        self.policy_loss = -policy_loss.mean()

        next_action    = self.target_policy_net(next_state)
        target_value   = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.hparams.gamma * target_value
        # NOTE can uncomment this
        # min_value=-np.inf;max_value=np.inf,
        # expected_value = torch.clamp(expected_value, min_value, max_value)

        value = self.value_net(state, action)
        self.value_loss = nn.MSELoss()(value.requires_grad_(True), expected_value.detach())


    def make_env(self):
        if self.env_type == 'linear':
            env = gym.make(self.env_name)
            env = NormalizedActions(env) # normalize actions
        # if self.env_type == 'cnn':
        #     env = make_atari(self.env_name)
        #     env = wrap_deepmind(env)
        #     env = wrap_pytorch(env)
        #     env = NormalizedActions(env) # normalize actions
        return env

    def populate(self, steps: int = 1000) -> None:
        """
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences
        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            self.agent.play_step(self.policy_net, epsilon=1.0)

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

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0].device.index if self.on_gpu else 'cpu'

    def get_epsilon(self):
        return max(self.hparams.eps_end, self.hparams.eps_start -
                      self.global_step + 1 / self.hparams.eps_last_frame)

    def run_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        """
            Run env step in environment and store rewards
        """
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

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx, optimizer_idx):
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
            self.loss(batch)
            loss = self.policy_loss
        if optimizer_idx == 1:
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
        value_optimizer = optimizers[self.hparams.opt](self.value_net.parameters(), lr=self.value_lr, weight_decay=self.hparams.weight_decay)
        policy_optimizer = optimizers[self.hparams.opt](self.policy_net.parameters(), lr=self.policy_lr, weight_decay=self.hparams.weight_decay)
        return [policy_optimizer, value_optimizer], []

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
            action = self.policy_net.get_action(state)
            next_state, reward, done, _ = self.test_env.step(action)
            state = next_state
            total_reward += reward
        return total_reward

    def train_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
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
    # token = telegram_config['token']
    # user_id = telegram_config['user_id']
    # bot = RLBot(token=token, user_id=user_id)
    # telegramCallback = TelegramRLCallback(bot)
    trainer = Trainer(checkpoint_callback=checkpoint_callback,
                    max_epochs=10000,
                    early_stop_callback=False,
                    val_check_interval=100,
                    logger=mlf_logger,
                    accumulate_grad_batches=1
    )
    model = myModule(hparams)
    trainer.fit(model)


if __name__ == '__main__':
    # torch.manual_seed(0)
    # np.random.seed(0)
    parser = argparse.ArgumentParser()
    # algo
    parser.add_argument("--algo_name", type=str, default='myModule',
                        help='Name of RL algorithm')
    parser.add_argument("--soft_tau", type=float, default=1e-2, help="weight for soft update of networks")
    parser.add_argument("--init_w", type=float, default=1e-3, help="weight initialization limits for networks")
    # parser.add_argument("--noisy", type=int, default=0, help="Whether to use noisy networks or not")
    # environment
    parser.add_argument("--env_name", type=str, default="Pendulum-v0", help="gym environment tag")
    parser.add_argument('--env_type', type=str, default='linear', choices=['linear', 'cnn'], help= 'type of network to use')
    parser.add_argument("--episode_length", type=int, default=1000, help="max length of an episode")
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
    parser.add_argument("--eps_last_frame", type=int, default=50000, help="what frame should epsilon stop decaying")
    parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
    parser.add_argument("--eps_end", type=float, default=0.01, help="final value of epsilon")
    # neural network
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--value_lr", type=float, default=1e-3, help="learning rate for value network")
    parser.add_argument("--policy_lr", type=float, default=1e-4, help="learning rate for policy network")
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid', 'tanh'], help='activations for nn layers')
    # optimizer
    parser.add_argument('--opt', type=str, default='adam', choices=['adam', 'adamax', 'rmsprop'], help='optimizer type for optimization')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay in optimizer')
    args = parser.parse_args()

    main(args)
