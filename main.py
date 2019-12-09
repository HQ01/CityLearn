import argparse
import torch
import time
import os
from pathlib import Path
import matplotlib as plt
import numpy as np
from maddpg import MA_DDPG
from citylearn import CityLearn
from buffer import ReplayBuffer
from reward_function import reward_function
from torch.utils.tensorboard import SummaryWriter
from citylearn import TIME_PERIOD

# implement main
USE_CUDA = False

# TODO: error when predefined action space does not match the actual space returned by env
# Building 9 raises this bug

def run(config):
    data_folder = Path(config.data_path)
    building_attributes = data_folder / 'building_attributes.json'
    solar_profile = data_folder / 'solar_generation_1kW.csv'
    building_state_actions = 'buildings_state_action_space.json'
    # building_ids = ["Building_" + str(i) for i in range(1, config.num_buildings + 1)]
    config.num_buildings = 6
    
    
    # customized log directory
    hidden = config.hidden_dim
    lr = config.lr
    tau = config.tau
    gamma = config.gamma
    batch_size = config.batch_size
    buffer_length = config.buffer_length
    to_print = lambda x : str(x)
    log_path = "log"+"_hidden"+to_print(hidden)+"_lr"+to_print(lr)+"_tau"+to_print(tau)+"_gamma"+to_print(gamma)+\
                "_batch_size"+to_print(batch_size)+"_buffer_length"+to_print(buffer_length)+"_TIME_PERIOD_1008"+"/"
    
    logger  = SummaryWriter(log_dir=log_path, comment='fuck')
    # TODO fix here
    building_ids = ["Building_" + str(i) for i in [1, 2, 5, 6, 7, 8]] #[1,2,5,6,7,8]
    env = CityLearn(building_attributes, solar_profile, building_ids, buildings_states_actions=building_state_actions,
                    cost_function=['ramping', '1-load_factor', 'peak_to_valley_ratio', 'peak_demand',
                                   'net_electricity_consumption'])
    observations_spaces, actions_spaces = env.get_state_action_spaces()

    # Instantiating the control agent(s)
    if config.agent_alg == 'MADDPG':
        agents = MA_DDPG(observations_spaces, actions_spaces, hyper_params=vars(config))
    else:
        raise NotImplementedError

    k, c = 0, 0
    cost, cum_reward = {}, {}
    buffer = ReplayBuffer(max_steps=config.buffer_length, num_agents=config.num_buildings,
                          obs_dims=[s.shape[0] for s in observations_spaces],
                          ac_dims=[a.shape[0] for a in actions_spaces])
    # TODO: store np or tensor in buffer?
    start = time.time()
    for e in range(config.n_episodes):
        cum_reward[e] = 0
        rewards = []
        state = env.reset()
        statecast = lambda x: [torch.FloatTensor(s) for s in x]
        done = False
        ss = 0
        while not done:
            if k % (40000 * 4) == 0:
                print('hour: ' + str(k) + ' of ' + str(TIME_PERIOD * config.n_episodes))
            action = agents.select_action(statecast(state))
            action = [a.detach().numpy() for a in action]
            # if batch norm:
            action = [np.squeeze(a, axis=0) for a in action]
            ss += 1
            #print("action is ", action)
            #print(action[0].shape)
            #raise NotImplementedError
            next_state, reward, done, _ = env.step(action)
            reward = reward_function(reward)  # See comments in reward_function.py
            #buffer_reward = [-r for r in reward]
            # agents.add_to_buffer()
            buffer.push(statecast(state), action, reward, statecast(next_state), done)
            # if (len(buffer) >= config.batch_size and
            #         (e % config.steps_per_update) < config.n_rollout_threads):
            if len(buffer) >= config.batch_size:
                if USE_CUDA:
                    agents.to_train(device='gpu')
                else:
                    agents.to_train(device='cpu')
                for a_i in range(agents.n_buildings):
                    sample = buffer.sample(config.batch_size,
                                           to_gpu=USE_CUDA)
                    agents.update(sample, a_i, logger=logger, global_step=e*TIME_PERIOD + ss)
            logger.add_scalar(tag='net electric consumption', scalar_value=env.net_electric_consumption[-1],
                              global_step=e*TIME_PERIOD + ss)
            logger.add_scalar(tag='env cost total', scalar_value=env.cost()['total'],
                              global_step=e*TIME_PERIOD + ss)
            logger.add_scalar(tag="1 load factor", scalar_value=env.cost()['1-load_factor'],
                              global_step=e*TIME_PERIOD + ss)
            logger.add_scalar(tag="peak to valley ratio", scalar_value=env.cost()['peak_to_valley_ratio'],
                              global_step=e*TIME_PERIOD + ss)
            logger.add_scalar(tag="peak demand", scalar_value=env.cost()['peak_demand'],
                              global_step=e*TIME_PERIOD + ss)
            logger.add_scalar(tag="net energy consumption", scalar_value=env.cost()['net_electricity_consumption'],
                              global_step=e*TIME_PERIOD + ss)
            net_energy_consumption_wo_storage = env.net_electric_consumption[-1]+env.electric_generation[-1]-env.electric_consumption_cooling_storage[-1]-env.electric_consumption_dhw_storage[-1]
            logger.add_scalar(tag="net energy consumption without storage", scalar_value=net_energy_consumption_wo_storage,
                              global_step=e*TIME_PERIOD + ss)
            
            for id, r in enumerate(reward):
                logger.add_scalar(tag="agent {} reward ".format(id), scalar_value=r, global_step=e*TIME_PERIOD+ss)
            
            state = next_state
            cum_reward[e] += reward[0]
            k += 1
            cur_time = time.time()
            # print("average time : {}s/iteration at iteration {}".format((cur_time - start) / (60.0 * k), k))
        cost[e] = env.cost()
        if c % 1 == 0:
            print(cost[e])
        # add env total cost and reward logger
        logger.add_scalar(tag='env cost total final', scalar_value=env.cost()['total'],
                          global_step=e)
        logger.add_scalar(tag="1 load factor final", scalar_value=env.cost()['1-load_factor'],
                          global_step=e)
        logger.add_scalar(tag="peak to valley ratio final", scalar_value=env.cost()['peak_to_valley_ratio'],
                          global_step=e)
        logger.add_scalar(tag="peak demand final", scalar_value=env.cost()['peak_demand'],
                          global_step=e)
        logger.add_scalar(tag="net energy consumption final", scalar_value=env.cost()['net_electricity_consumption'],
                          global_step=e)
        net_energy_consumption_wo_storage = env.net_electric_consumption[-1]+env.electric_generation[-1]-env.electric_consumption_cooling_storage[-1]-env.electric_consumption_dhw_storage[-1]
        logger.add_scalar(tag="net energy consumption without storage", scalar_value=net_energy_consumption_wo_storage,
                          global_step=e)
        c += 1
        rewards.append(reward)

    end = time.time()
    print((end - start) / 60.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='data/', type=str,
                        help="Path of environment data")
    parser.add_argument("--log_path", default='log/', type=str,
                        help="Path of log files")
    parser.add_argument("--num_buildings", default=2, type=int,
                        choices=range(1, 10), help="Number of buildings (1 to 9)")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    # parser.add_argument("--n_rollout_threads", default=1, type=int)
    # parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=2000, type=int)
    parser.add_argument("--n_episodes", default=25000, type=int)
    # parser.add_argument("--episode_length", default=25, type=int)
    # parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=32, type=int,
                        help="Batch size for model training")
    # parser.add_argument("--n_exploration_eps", default=25000, type=int)
    # parser.add_argument("--init_noise_scale", default=0.3, type=float)
    # parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=32, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--tau", default=1e-6, type=float)
    parser.add_argument("--gamma", default=0.992, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    # parser.add_argument("--adversary_alg",
    #                     default="MADDPG", type=str,
    #                     choices=['MADDPG', 'DDPG'])
    # parser.add_argument("--discrete_action",
    #                     action='store_true')

    config = parser.parse_args()

    run(config)
