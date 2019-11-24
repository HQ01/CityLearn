import torch
import torch.nn.functional as fix
from gym.spaces import Box, Discrete
from misc import soft_update, average_gradients
from single_agents import DDPGAgent



class MA_DDPG():
    def __init__(self, observation_spaces = None, action_spaces = None, hyper_params = None):
        """
        Input:
            observation_spaces
            action_spaces
            hyper_params
        """
        self.n_buildings = len(observation_spaces)
        self.agents = [DDPGAgent(#\TODO
                        )]
        #\TODO fix init param
        self.gamma = None
        self.tau = None
        self.lr = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    @property
    def policies(self):
        return [a.actor for a in self.agents]
    
    @property
    def target_policies(self):
        return [a.actor_target for a in self.agents]
    
    def scale_noise(self, scale): #\TODO do we need this
        for a in self.agents:
            a.scale_noise(scale)
    
    def reset_noise(self): #\TODO do we need this
        for a in self.agents:
            a.reset_noise()
    
    def select_action(self, observations, explore=False):
        '''
        Take a step of actions for all agents)
        '''
        return [a.select_action(obs, explore=explore) for a, obs in zip(self.agents, observations)]
    
    def update(self, sample, agent_id, logger=None):
        obs, actions, rewards, next_obs, dones = sample # notice that we still need dones because sample could be at different place

        curr_agent = self.agents[agent_id]
        curr_agent.critic_optimizer.zero_grad()
        all_target_actions = [policy(next_ob) for policy, next_ob in zip(self.target_policies, next_obs)]
        target_value_function = [policy(next_ob) for policy, next_ob in zip(self.target_policies, next_obs)]

        target_value = (rewards[agent_id].view(-1, 1) + self.gamma * curr_agent.critic_target(target_value_function) *\
             (1 - dones[agent_id].view(-1, 1)))
        
        
        actual_value_function = torch.cat((*obs, *actions), dim=1)

