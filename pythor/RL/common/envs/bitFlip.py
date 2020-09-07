"""
    Goal based environment for flipping bits
    Use for HER
"""
import numpy as np

class Env(object):
    def __init__(self, num_bits):
        self.num_bits = num_bits
    
    def reset(self):
        self.done      = False
        self.num_steps = 0
        self.state     = np.random.randint(2, size=self.num_bits)
        self.target    = np.random.randint(2, size=self.num_bits)
        return self.state, self.target
    
    def step(self, action):
        if self.done:
            raise RESET
        
        self.state[action] = 1 - self.state[action]
        
        if self.num_steps > self.num_bits + 1:
            self.done = True
        self.num_steps += 1
        
        if np.sum(self.state == self.target) == self.num_bits:
            self.done = True
            return np.copy(self.state), 0, self.done, {}
        else:
            return np.copy(self.state), -1, self.done, {}