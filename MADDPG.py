# --------------------------------------
# Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
# Author: Adrian Chow
# Date: 2020.1.16
# Reference: https://papers.nips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf
# Purpose: Adaption for Multi-Agent Environments for DDPG
# --------------------------------------

import numpy as np
from collections import namedtuple, deque
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from OUNoise import OrnsteinUhlenbeckNoise
from Networks import Actor, Critic
from Buffer import ReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 250        # minibatch size
GAMMA = 0.99           # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.       # L2 weight decay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MADDPG:
    """ Class for training multiple agents in the multi-agent environment"""

    def __init__(self, state_size, action_size, num_agents, seed):
        
        super(MADDPG, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(seed)

        # Initialize agents in the multi-agent environment
        self.DDPGs = [DDPG(state_size, action_size, num_agents, seed) for i in range(num_agents)]
        
        # Replay Buffer (shared by all agents)
        self.memory = ReplayBuffer(action_size, num_agents, BUFFER_SIZE, BATCH_SIZE, seed, device)


    def act(self, states, add_noise=True):
        """Returns actions for each agent."""

        return [agent.act(state, add_noise) for agent, state in zip(self.DDPGs, states)]


    def step(self, states, actions, rewards, next_states, dones):

        self.memory.add(states, actions, rewards, next_states, dones)
        
        for agent in self.DDPGs:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                agent.step(experiences, GAMMA)
    

    def reset(self):
        """ Reset Agents """
        for agent in self.DDPGs:
            agent.reset()        


    def save_weights(self):
        for index, agent in enumerate(self.DDPGs):
            torch.save(agent.actor.state_dict(), 'agent{}_checkpoint_actor.pth'.format(index+1))
            torch.save(agent.critic.state_dict(), 'agent{}_checkpoint_critic.pth'.format(index+1))


class DDPG:
    """ Base Class for an Agent in the multi-agent environment"""

    def __init__(self, 
                 state_size, 
                 action_size,
                 num_agents,
                 seed, 
                 hidden_in_actor=200,
                 hidden_out_actor=150,
                 hidden_in_critic=200, 
                 hidden_out_critic=150, 
                 lr_actor=LR_ACTOR, 
                 lr_critic=LR_CRITIC):

        super(DDPG, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(seed)
        
        self.actor = Actor(state_size, 
                           action_size,
                           seed,
                           hidden_in_actor,
                           hidden_out_actor).to(device)

        self.target_actor = Actor(state_size, 
                                  action_size,
                                  seed,
                                  hidden_in_actor,
                                  hidden_out_actor).to(device)

        self.critic = Critic(state_size, 
                             action_size,
                             num_agents,
                             seed,
                             hidden_in_critic,
                             hidden_out_critic).to(device)

        self.target_critic = Critic(state_size, 
                                    action_size,
                                    num_agents,
                                    seed,
                                    hidden_in_critic,
                                    hidden_out_critic).to(device)
        
        # Ornstein Uhlenbeck Noise for Action Space Exploration
        self.noise = OrnsteinUhlenbeckNoise(action_size, seed) 

        # Initialize targets same as original networks
        self.copy_weights(self.critic, self.target_critic)
        self.copy_weights(self.actor, self.target_actor)

        # Actor and Critic Adam Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)


    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        
        state = torch.from_numpy(state).float().to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        if add_noise:
            action += self.noise.noise()
        return np.clip(action, -1, 1)


    def step(self, experiences, gamma):
        self.learn(experiences, gamma)


    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * target_critic(next_state, target_actor(next_state))
        where:
            target_actor(state) -> action
            target_critic(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', a2, s2' done) tuples 
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences

        next_states_tensor = torch.cat(next_states, dim=1).to(device)
        states_tensor = torch.cat(states, dim=1).to(device)
        actions_tensor = torch.cat(actions, dim=1).to(device)        

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        
        next_actions = [self.actor(state) for state in states]        
        next_actions_tensor = torch.cat(next_actions, dim=1).to(device)      
        Q_targets_next = self.target_critic(next_states_tensor, next_actions_tensor)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic(states_tensor, actions_tensor)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = [self.actor(state) for state in states]        
        actions_pred_tensor = torch.cat(actions_pred, dim=1).to(device)
        actor_loss = -self.critic(states_tensor, actions_pred_tensor).mean()
        # Minimize the loss, thererby maximizing the reward
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic, self.target_critic, TAU)
        self.soft_update(self.actor, self.target_actor, TAU)                     


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def reset(self):
        self.noise.reset()    


    def copy_weights(self, source, target):
        """Copies the weights from the source to the target"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)
