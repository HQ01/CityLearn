import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy

#random seed
np.random.seed(1)

class Actor_TD3(nn.Module):   
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor_TD3, self).__init__()

        self.l1 = nn.Linear(state_dim, 5)
        self.l2 = nn.Linear(5, 3)
        self.l3 = nn.Linear(3, action_dim)

        self.max_action = max_action
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic_TD3(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic_TD3, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 7)
        self.l2 = nn.Linear(7, 6)
        self.l3 = nn.Linear(6, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 7)
        self.l5 = nn.Linear(7, 6)
        self.l6 = nn.Linear(6, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class TD3_single():
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip = 0.5, policy_freq=2):
        # default params follows https://github.com/sfujim/TD3/blob/master/TD3.py
        
        self.actor = Actor_TD3(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0