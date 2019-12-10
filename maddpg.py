import torch
import torch.nn.functional as fix
# from gym.spaces import Box, Discrete
# from misc import soft_update, average_gradients
from single_agents import DDPG_single, TD3_single
from utils import ActionSpaceConverter

MSELoss = torch.nn.MSELoss()
class MA_DDPG():
    def __init__(self, observation_spaces = None, action_spaces = None, hyper_params = {}, discrete_action=True, grid_per_action=20):
        """
        Input:
            observation_spaces
            action_spaces
            hyper_params
        """
        self.n_buildings = len(observation_spaces)
        self.algo_type = 'DDPG' # "TD3"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.discrete_action = discrete_action
        self.grid_per_action = grid_per_action

        self.gamma = hyper_params.get('gamma', 0.992) #self.discount
        self.lr = hyper_params.get('lr', 1e-4)
        # self.batch_size = hyper_params.get('batch_size', 100)
        self.tau = hyper_params.get('tau', 5e-3)
        #\TODO should we enable min_samples_training?
        # self.min_samples_training = hyper_params.get('min_samples_training', 400)
        self.max_action = hyper_params.get('max_action', 0.25)
        self.hidden_dim = hyper_params.get("hidden_dim", 32)

        '''
        # we don't change lr rate at this time \TODO verify s
        self.lr_init = hyper_params.get('lr_init', 1e-3)
        self.lr_final = hyper_params.get('lr_final', 1e-3)
        self.lr_decay_rate = hyper_params.get('lr_decay_rate', 1 / (78 * 8760))
        '''


        # TD3 hyper-params
        self.policy_freq = hyper_params.get('policy_freq', 10)
        self.policy_noise = hyper_params.get('policy_noise', 0.025 * 0)
        self.noise_clip = hyper_params.get('noise_clip', 0.04 * 0)
        self.expl_noise_init = hyper_params.get('expl_noise_init', 0.75)
        self.expl_noise_final = hyper_params.get('expl_noise_final', 0.01)
        # Decay rate of the exploration noise in 1/h
        self.expl_noise_decay_rate = hyper_params.get('expl_noise_decay_rate', 1/(290*8760))


        # Monitoring variables (one per agents)
        #\TODO add monitoring support


        # init agents
        #\TODO match dimension with buffer input
        state_dim = observation_spaces[0].shape[0]
        action_dim = action_spaces[0].shape[0]
        max_action = action_spaces[0].high
        converter = ActionSpaceConverter(max_action, self.grid_per_action)

        if self.algo_type == 'DDPG':
            self.agents = [DDPG_single(state_dim, action_dim, self.max_action, num_agents=self.n_buildings, learning_rate=self.lr, discrete_action = self.discrete_action, grid_per_action=self.grid_per_action, hidden_dim=self.hidden_dim)]
        else:
            self.agents = [TD3_single(state_dim, action_dim, self.max_action, self.expl_noise_init, self.expl_noise_final, self.expl_noise_decay_rate)]
        self.agents *= self.n_buildings

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
    
    def reset_iter(self):
        for a in self.agents:
            a.iter = 0
    
    def select_action(self, observations, explore=True):
        '''
        Take a step of actions for all agents)
        '''
        return [a.select_action(obs, explore=explore) for a, obs in zip(self.agents, observations)]
    
    def update(self, sample, agent_id, logger=None, global_step=None):
        obs, actions, rewards, next_obs, dones = sample # notice that we still need dones because sample could be at different place

        curr_agent = self.agents[agent_id]
        
        # update critic
        curr_agent.critic_optimizer.zero_grad()
        
        all_target_actions = [policy(next_ob) for policy, next_ob in zip(self.target_policies, next_obs)]

        # section 5.3 equation (14)
        if self.algo_type == 'TD3':
            noise_f = lambda x : x + (torch.randn_like(x) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip).to(x.device)
            all_target_actions = list(map(noise_f, all_target_actions))

        target_value_function = torch.cat((*next_obs, *all_target_actions), dim=1)
        target_value = (
                rewards[agent_id].view(-1, 1) + self.gamma * curr_agent.critic_target(target_value_function) * \
                (1 - dones[agent_id].view(-1, 1)))
        # print(rewards[agent_id])
        actual_value_function = torch.cat((*obs, *actions), dim=1)
        actual_value = curr_agent.critic(actual_value_function)

        value_function_loss = MSELoss(actual_value, target_value.detach())
        # print([a.detach().numpy() for a in [value_function_loss.mean(), actual_value.mean()]])
        value_function_loss.backward()
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        if self.algo_type == "DDPG" or curr_agent.iter % self.policy_freq == 0:
            #update actor
            #if TD3: delayed policy updates, section 5.2 
            curr_agent.actor_optimizer.zero_grad()
            all_agent_policies = []
            curr_agent_policy = curr_agent.actor(obs[agent_id])
            for i, policy, ob in zip(range(self.n_buildings), self.policies, obs):
                if i == agent_id:
                    all_agent_policies.append(curr_agent_policy)
                else:
                    all_agent_policies.append(policy(ob))

            policy_function = torch.cat((*obs, *all_agent_policies), dim=1)

            policy_function_loss = -curr_agent.critic(policy_function).mean()
            policy_function_loss += (curr_agent_policy**2).mean() * 1e-3
            policy_function_loss.backward()
            # print(self.agents[0].actor.l1.weight.grad.mean(), self.agents[0].actor.l1.weight.grad.sum(), end=' ')
            # print(self.agents[0].actor.l1.weight.grad.norm())

            # torch.nn.utils.clip_grad_norm_(curr_agent.actor.parameters(), 0.5)
            curr_agent.actor_optimizer.step()

            # update target
            self.soft_update(curr_agent.actor_target, curr_agent.actor, self.tau)
            self.soft_update(curr_agent.critic_target, curr_agent.critic, self.tau)

        curr_agent.iter += 1
        if logger is not None:
            logger.add_scalar(tag='agent %d value function loss' % (agent_id), scalar_value=value_function_loss,
                              global_step=global_step)
            logger.add_scalar(tag='agent %d policy function loss' % (agent_id), scalar_value=value_function_loss,
                              global_step=global_step)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    # def update_targets(self):
    #     for a in self.agents:
    #         self.soft_update(a.actor_target, a.actor,delayedf.tau)
    #         self.soft_update(a.critic_target, a.critic, self.tau)
    
    def to_train(self, device = None):
        if not device: device = self.device
        for a in self.agents:
            a.actor.train()
            a.critic.train()
            a.actor_target.train()
            a.critic_target.train()
        self.to_device(device)
    
    def to_eval(self):
        for a in self.agents:
            a.actor.eval()
            a.actor.to('cpu')
            
    
    def to_device(self, device=None):
        if not device: device = self.device
        for a in self.agents:
            a.actor.to(device)
            a.critic.to(device)
            a.actor_target.to(device)
            a.critic_target.to(device)
    
    def save_model(self, filename):
        #\TODO decide what to store
        self.to_train('cpu')
        hyper_params = self.pack_hyper_params()
        save_dict = {'hyper_params': hyper_params, "weights": [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    def pack_hyper_params(self):
        #\TODO decide what to pack
        # hyper_params = {
        #     'gamma' : self.gadelayed
        #     'tau' : self.tau,delayed
        #     "lr" : self.lr_init,
        #
        # }

        return hyper_params
    
    

        




