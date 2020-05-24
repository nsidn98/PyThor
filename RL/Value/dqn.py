'''
Deep Q Networks
'''
import os
import math, random
import gym
import numpy as np
import datetime
from tqdm import tqdm

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
from pytorch_lightning import loggers


from torchvision import datasets, transforms
from torchvision import transforms

class DQN(nn.Module):
    """
        Network for DQN function approximation
        Parameters:
        -----------
        input_shape : int
            State vector dimension
        num_actions : int
            Number of actions
        Can be used for OpenAI gym type environments like CartPole-v0, etc.
    """
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon=0):
        """
            Give actions for given states
            Parameters:
            -----------
            state: numpy array of shape [input_shape,]
                State for the RL agent
                NOTE: the state will be converted torch tensor and will be unsqueezed 
                to account for batch_size
            epsilon: float in range [0,1]
                For epsilon-greedy (default value is 0)
        """
        if random.random() > epsilon:
            state   = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_value = self.forward(state)
            action  = int(q_value.max(1)[1].data[0])
        else:
            action = random.randrange(self.num_actions)
        return action

class CnnDQN(nn.Module):
    """
        Network for DQN function approximation
        where states are images
        Parameters:
        -----------
        input_shape : list [c,h,w]
            State vector dimension 
            Example: [3,28,28]
        num_actions : int
            Number of actions
        Can be used for OpenAI gym-Atari type environments like PongNoFrameskip-v4, etc.
        NOTE: will have to use 
            from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
            env_id = "PongNoFrameskip-v4"
            env    = make_atari(env_id)
            env    = wrap_deepmind(env)
            env    = wrap_pytorch(env)
        for environment to be compatible
    """
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, env, epsilon=0):
        """
            Give actions for given states
            Parameters:
            -----------
            state: numpy array of shape [c,h,w,]
                State for the RL agent
                NOTE: the state will be converted torch tensor and will be unsqueezed 
                to account for batch_size
            epsilon: float in range [0,1]
                For epsilon-greedy (default value is 0)
        """
        if random.random() > epsilon:
            state   = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(device)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.num_actions)
        return action

class Agent:
    """
    Base Agent class handeling the interaction with the environment
    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state = self.env.reset()

    def reset(self) -> None:
        """ Resents the environment and updates the state"""
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
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
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(self, net: nn.Module, epsilon: float = 0.0, device: str = 'cpu') -> Tuple[float, bool]:
        """
        Carries out a single interaction step between the agent and the environment
        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device
        Returns:
            reward, done
        """

        action = self.get_action(net, epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)

        exp = Experience(self.state, action, reward, done, new_state)

        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done

class DQN_Agent(LightningModule):
    def __init__(self, env, test_env, model_type, warm_start_steps):
        self.env = env
        self.test_env = test_env
        if model_type == 'linear':
            self.current_model = DQN(self.env.observation_space.shape[0], self.env.action_space.n)
        elif model_type == 'cnn':
            obs_shape = self.env.observation_space.shape # example (480,280,3)
            obs_shape = np.roll(np.array(obs_shape),1)
            self.current_model = CnnDQN(obs_shape, self.env.action_space.n)

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.populate(warm_start_steps)

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
        output = self.current_model(x)
        return output

class DQN_Agent:
    def __init__(self, env, test_env, save_name, args):
        self.args = args
        self.save_name = save_name
        self.env = env
        self.test_env = test_env
        self.env.seed(self.args.seed)
        self.test_env.seed(self.args.seed)
        if self.args.CNN:
            self.current_model = CnnDQN(self.env.observation_space.shape, self.env.action_space.n).to(device)
            self.buffer_size = 100000
            self.replay_initial = 10000
        else:    
            self.current_model = DQN(self.env.observation_space.shape[0], self.env.action_space.n).to(device)
            self.buffer_size = 1000
            self.replay_initial = self.args.batch_size # NOTE: currently setting it to batch size. Can increase it to some higher value like 100
        self.optimizer = optim.Adam(self.current_model.parameters(), lr = self.args.lr)
        self.replay_buffer = ReplayBuffer(self.buffer_size)  # buffer with original environment rewards
        # seeds
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        # tensorboardx
        if self.args.tensorboard:
                # print('Init tensorboardX')
                # self.writer = SummaryWriter(log_dir='runs/{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
                self.writer = SummaryWriter(log_dir='runs/'+save_name)

    
    def compute_td_loss(self, batch, grad = True):
        '''
        Compute the loss for the Q-networks
        '''

        state, action, reward, next_state, done = batch

        state      = Variable(torch.FloatTensor(np.float32(state)),requires_grad = grad).to(device)
        next_state = Variable(torch.FloatTensor(np.float32(next_state)),requires_grad= grad).to(device)
        action     = Variable(torch.LongTensor(action)).to(device)
        reward     = Variable(torch.FloatTensor(reward)).to(device)
        done       = Variable(torch.FloatTensor(done)).to(device)

        q_values      = self.current_model(state)
        next_q_values = self.current_model(next_state)

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + self.args.gamma * next_q_value * (1 - done)
        
        loss = (q_value - (expected_q_value.data)).pow(2).mean()

        # evaluate gradient of loss wrt inputs for evaluating aux_rewards
        if grad:
            gradient = torch.autograd.grad(loss,state)
        else:
            gradient = None
        
        return loss, gradient


    def test_episode(self,step):
        state = self.test_env.reset()
        episode_reward = 0
        while True:
            action = self.current_model.act(state,env=self.test_env,epsilon=0)
            next_state, reward, done, _ = self.test_env.step(action)
            state = next_state
            episode_reward += reward
            if done:
                if self.args.tensorboard:
                    self.writer.add_scalar('Reward',episode_reward,step)
                break
        return episode_reward

        

    def train(self):
        frame_idx = 0
        episode_reward = 0
        episode_num = 0
        episode_reward_buffer = []

        test_step = 0

        state = self.env.reset()

        for frame_idx in tqdm(range(self.args.max_steps)):
            # frame_idx += 1
            if not self.args.grad_explore:
                epsilon = epsilon_by_frame(frame_idx)
            else:
                epsilon = 0
            action = self.current_model.act(state, env=self.env, epsilon=epsilon)
            action = self.current_model.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if done:
                episode_num += 1
                state = self.env.reset()
                # episode_reward_buffer.append(episode_reward)
                # if self.args.tensorboard:
                #     self.writer.add_scalar('Reward',episode_reward,episode_num)
                episode_reward = 0

            if frame_idx % 100 == 0:
                test_rew = self.test_episode(test_step) # reward for one episode after 'test_step' timesteps of training
                test_step += 1
                episode_reward_buffer.append(test_rew)
            
            if len(self.replay_buffer) > self.replay_initial:
                batch = self.replay_buffer.sample(self.args.batch_size)
                loss, gradient = self.compute_td_loss(batch, grad = self.args.grad_explore)
                # if gradient exploration method, then compute aux_rewards and the new loss with new rewards
                if self.args.grad_explore:
                    loss = self.compute_td_loss_aux_rewards(batch,gradient,frame_idx)

                # backward prop and update weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # log the loss
                if self.args.tensorboard:
                    self.writer.add_scalar('Loss',loss.item(),frame_idx)
            
            # if len(episode_reward_buffer) > 100:
            #     # solved criteria for the environment
            #     if np.mean(np.array(episode_reward_buffer[-100:])) > self.args.env_max_rew or episode_num == self.args.max_episodes:
            #     # if episode_num == 1000:
            #         np.save(self.save_name +'.npy',np.array(episode_reward_buffer))
            #         break
        np.save('Exps/'+self.save_name +'.npy',np.array(episode_reward_buffer))