import numpy as np
import math, random
import collections
from torch.utils.data import IterableDataset


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

class PrioritizedBuffer:
    def __init__(self, capacity: int, prob_alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 1000):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
        # self.buffer = []
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.update_beta(0)
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def append(self,  experience: Experience):
        max_prior = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        self.priorities[self.pos] = max_prior
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool)
        next_states = np.array(next_states)

        return states, actions, rewards, dones, next_states, indices, weights

    def __len__(self):
        return len(self.buffer)

    def update_beta(self,step:int):
        self.beta = min(1.0, self.beta_start + step * (1.0 - self.beta_start) / self.beta_frames)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prior in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prior

