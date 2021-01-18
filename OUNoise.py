# --------------------------------------
# Ornstein-Uhlenbeck Noise
# Author: Adrian Chow
# Date: 2020.1.16
# Reference: https://github.com/songrotek/DDPG/blob/master/ou_noise.py
# Purpose: Provide Stoichastic Noise for Continuous Action Exploration
# --------------------------------------

import torch
import numpy as np
import random

class OrnsteinUhlenbeckNoise: 
    ''' Provide Stoichastic Noise for Continuous Action Exploration '''
    
    def __init__(self, action_size, seed, scale=1.0, mu=0.0, theta=0.15, sigma=0.2):
        
        self.action_size = action_size
        self.scale = scale
        self.seed = np.random.seed(seed)
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_size) * self.mu
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_size) * self.mu
        
    def noise(self):
        
        # Noise State
        x = self.state
        
        # Weiner Process
        weiner_process = np.random.randn(self.action_size)
       
        # Deriving The Noise State
        dx = self.theta * (self.mu - x) + self.sigma * weiner_process
        
        # Update Noise State
        self.state = x + dx
        
        # Sacle and Convert to Float Tensor
        return (self.state)
