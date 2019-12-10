# CityLearn with MADDPG
This is the repository for CS 394R Reinforcement Learnin Term project by Tong Gao and Qi Huang. It implemented MADDPG (Multi-agent deep deterministic policy gradient) on CityLearn. It's based on the CityLearn
environment by Vázquez-Canteli et.al. at UT Austin intelligent environment lab (https://github.com/intelligent-environments-lab/CityLearn)

# CityLearn
CityLearn is an open source OpenAI Gym environment for the implementation of Multi-Agent Reinforcement Learning (RL) for building energy coordination and demand response in cities. Its objective is to facilitiate and standardize the evaluation of RL agents such that different algorithms can be easily compared with each other.
![Demand-response](https://github.com/intelligent-environments-lab/CityLearn/blob/master/images/dr.jpg)
## Files
- [main.py](/main.py): python script file. Training script for MADDPG on CityLearn
- [buildings_state_action_space.json](/buildings_state_action_space.json): json file containing the possible states and actions for every building, from which users can choose.
- [building_attributes.json](/data/building_attributes.json): json file containing the attributes of the buildings and which users can modify.
- [citylearn.py](/citylearn.py): Contains the ```CityLearn``` environment and the functions ```building_loader()``` and ```autosize()```
- [energy_models.py](/energy_models.py): Contains the classes ```Building```, ```HeatPump``` and ```EnergyStorage```, which are called by the ```CityLearn``` class
- [single_agents.py](/single_agents.py): Implementation of the Deep Deterministic Policy Gradient ([DDPG](https://arxiv.org/abs/1509.02971)) RL algorithm as an single agent.
- [maddpg.py](/maddpg.py): Implementation of the maddpg algorithm in PyTorch. We refer to (https://github.com/shariqiqbal2810/maddpg-pytorch) and the TensorFlow official implementation (https://github.com/openai/maddpg) during our work.
- [reward_function.py](/reward_function.py): Contains the reward function that wraps and modifies the rewards obtained from ```CityLearn```. This function can be modified by the user in order to minimize the cost function of ```CityLearn```.

## To Run
To run the code:
``python main.py``

### Arguments
- `--data_path`: building simulation data path (default: `data/`)
- `--log_path`: training log save path (default: `log/`)
- `--num_buildings`: number of buildings(agents) (defualt: `6`)
- `--lr`: learning rate (default: `1e-4`)
- `--gamma`: discount factor (default: `0.992`)
- `--batch_size`: batch size (default: `32`)
- `--hidden_dim`: hidden dimension (default: `32`)
- `--tau`: target actor/critic update factor (defualt: `1e-6`)
- `agent_alg`: agent's algorithm (default: `MADDPG`)
- `buffer_length`: maximal muffer length (default: `2000`)
- `n_episodes`: number of episodes (default: `25000`)

To change episode length, go to [citylearn.py](/citylearn.py) and change `TIME_PERIOD`
## License
The MIT License (MIT) Copyright (c) 2019, José Ramón Vázquez-Canteli
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
