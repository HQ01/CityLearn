import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from utils import OUNoise

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
        # section 4.2 eq (10)
        return torch.min(q1, q2)

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class Actor_DDPG(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor_DDPG, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action

	
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic_DDPG(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic_DDPG, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)


	def forward(self, state, action):
		q = F.relu(self.l1(torch.cat([state, action], 1)))
		q = F.relu(self.l2(q))
		return self.l3(q)

class TD3_single():
    def __init__(self, state_dim, action_dim, max_action, expl_noise_init, expl_noise_final, expl_noise_decay_rate):
        # default params follows https://github.com/sfujim/TD3/blob/master/TD3.py
        
        self.max_action = max_action
        self.expl_noise_init = expl_noise_init
        self.expl_noise_final = expl_noise_final
        self.expl_noise_decay_rate = expl_noise_decay_rate
        
        self.actor = Actor_TD3(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic_TD3(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		
        # counter
        self.iter = 0
    
    def select_action(self, obs):
        expl_noise = max(self.expl_noise_final, self.expl_noise_init * (1 - self.iter * self.expl_noise_decay_rate))
        action = self.actor(obs) # assume they are on the same device
        action += torch.Tensor(expl_noise * np.random.normal(loc = 0, scale = self.max_action, size = action.size())).to(action.device())
        action = action.clamp(-self.max_action, self.max_action)

        return action
    

class DDPG_single():
    def __init__(self, state_dim, action_dim, max_action):
        self.max_action = max_action
        
        self.actor = Actor_DDPG(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic_DDPG(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.exploration = OUNoise(action_dim)

        self.iter = 0

    
    def scale_noise(self, scale):
        self.exploration.scale = scale
    
    def reset_noise(self):
        self.exploration.reset()
    
    def select_action(self, obs, explore=False):
        action = self.actor(obs)
        if explore:
            device = action.device()
            action += torch.Tensor(self.exploration.noise()).to(device)
        action = action.clamp(-self.max_action, self.max_action)

        return action
    
    def get_params(self):
        return {'actor' : self.actor.state_dict(),
                'actor_target' : self.actor_target.state_dict(),
                'critic' : self.critic.state_dict(),
                'critic_target' : self.critic_target.state_dict(),
                'actor_optimizer' : self.actor_optimizer.state_dict(),
                'critic_optimizer' : self.critic_optimizer.state_dict()
                }
    
    def load_params(self, params):
        self.actor.load_state_dict(params['actor'])
        self.actor_target.load_state_dict(params['actor_target'])
        self.actor_optimizer.load_state_dict(params['actor_optimizer'])

        self.critic.load_state_dict(params['critic'])
        self.critic_target.load_state_dict(params['critic_target'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])