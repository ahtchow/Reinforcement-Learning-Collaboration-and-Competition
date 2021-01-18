# --------------------------------------
# Replay Buffer
# Author: Adrian Chow
# Date: 2020.1.16
# Purpose: Provide Data Structure for Sampling Experiences without Bias
# --------------------------------------

from collections import namedtuple, deque
import numpy as np
import random
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

field_names = ["state", "action", "reward", "next_state", "done"]

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, num_agents, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """

        self.action_size = action_size,
        self.num_agents = num_agents
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=field_names)
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        # Sample an experience with length k from list of memories
        experiences = random.sample(self.memory, k=self.batch_size)

        states = [torch.from_numpy(np.vstack([e.state[index] for e in experiences if e is not None])).float().to(device) for index in range(self.num_agents)]
        actions = [torch.from_numpy(np.vstack([e.action[index] for e in experiences if e is not None])).float().to(device) for index in range(self.num_agents)]
        next_states = [torch.from_numpy(np.vstack([e.next_state[index] for e in experiences if e is not None])).float().to(device) for index in range(self.num_agents)]            
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)        
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
