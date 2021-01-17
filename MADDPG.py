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
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

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
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device)


    def act(self, states, add_noise=True):
        """Returns actions for each agent."""
        return [agent.act(state, add_noise) for agent, state in zip(self.DDPGs, states)]


    def step(self, states, actions, rewards, next_states, dones):

        self.memory.add(self.encode(states), 
                        self.encode(actions), 
                        self.encode(rewards),
                        self.encode(next_states),
                        self.encode(dones))

        if len(self.memory) > BATCH_SIZE:
            for i in range(len(self.DDPGs)):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA, agent_num=i)
    

    def learn(self, experiences, gamma, agent_num):
        """ Sample from Replay Buffer and update actor-critic weights """

        states, actions, rewards, next_states, dones = experiences
        
        # Decode
        state = self.decode(states, agent_num)
        action = self.decode(actions, agent_num)
        reward = self.decode(rewards, agent_num)
        next_state = self.decode(next_states, agent_num)
        done = self.decode(dones, agent_num)

        # Exclusive to Tennis environment
        opp_num = 1 if agent_num == 0 else 0
        opp_action = self.decode(actions, opp_num)
        opp_next_state = self.decode(next_states, opp_num)

        decoded_experiences = (state, action, reward, next_state, done, opp_action, opp_next_state)

        self.DDPGs[agent_num].learn(decoded_experiences, gamma)


    def encode(self, info):
        """ 
        For multi-agent experiences - encode required since each time step 
        holds mutiple states and actions for each given agent
        """

        # Flatten and Squeeze
        return np.array(info).reshape(1,-1).squeeze()


    def decode(self, info, agent_num):
        """ 
        For multi-agent experiences - decode required since each time step 
        holds mutiple states and actions for each given agent
        """
        info = info.detach().numpy().reshape(1,-1).squeeze() # Flatten
        info_decoded = torch.Tensor([np.split(data, self.num_agents)[agent_num] for data in info])
        return info_decoded

    def reset(self):
        """ Reset Agents """
        for agent in self.DDPGs:
            agent.reset()        



class DDPG:
    """ Base Class for an Agent in the multi-agent environment"""

    def __init__(self, 
                 state_size, 
                 action_size,
                 num_agents,
                 seed, 
                 hidden_in_actor=128,
                 hidden_out_actor=128,
                 hidden_in_critic=128, 
                 hidden_out_critic=128, 
                 lr_actor=LR_ACTOR, 
                 lr_critic=LR_CRITIC):

        super(DDPG, self).__init__()
        
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
        self.noise = OrnsteinUhlenbeckNoise(num_agents * action_size, seed, scale=1.0)

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


    def learn(self, experiences, gamma):
            """Update policy and value parameters using given batch of experience tuples.
            Q_targets = r + γ * target_critic(next_state, target_actor(next_state))
            where:
                target_actor(state) -> action
                target_critic(state, action) -> Q-value
            Params
            ======
                experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
                gamma (float): discount factor
            """
            states, actions, rewards, next_states, dones, opp_actions, opp_next_states = experiences

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            
            actions_next = self.target_actor(next_states)
            actions_next_other_player = self.target_actor(opp_next_states)
            Q_targets_next = self.target_critic(next_states, actions_next, actions_next_other_player)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            # Compute critic loss
            Q_expected = self.critic(states, actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor(states)
            actor_loss = -self.critic(states, actions_pred, opp_actions).mean()
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

    
