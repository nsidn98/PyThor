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

class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the ReplayBuffer
    This will be used in the lightning module for the agent
    for dataloder
    which will be updated with new experiences during training
    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]

# class ReplayBuffer(object):
#     """
#         Replay Buffer for storing 
#         states, actions, rewards, next_state and done (s,a,r,s,d)
#     """
#     def __init__(self, capacity):
#         """
#             Parameters:
#             -----------
#             capacity : int
#                 Capacity of the buffer to store (s,a,r,s,d) tuples
#                 The tuples will get dequed once the buffer gets full
#         """
#         self.buffer = deque(maxlen=capacity)
    
#     def push(self, state, action, reward, next_state, done):
#         """
#             Push the transitions
#         """
#         state      = np.expand_dims(state, 0)
#         next_state = np.expand_dims(next_state, 0)
#         self.buffer.append((state, action, reward, next_state, done))
    
#     def sample(self, batch_size):
#         """
#             Sample transitions
#         """
#         state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
#         return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
#     def __len__(self):
#         return len(self.buffer)