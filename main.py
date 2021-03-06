# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:59:07 2021

@author: Sarah Li
"""
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import ddpg as ddpg

import actor_critic_nn as nets
from chain_model import SupplyChain
import util as ut


use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
# env = NormalizedActions(gym.make("Pendulum-v0"))
env = SupplyChain()


state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.low.shape[0]
ou_noise = ut.OUNoise(env.action_space)

hidden_dim = 256

model = namedtuple('model', ['value_net', 'policy_net', 
         'target_value_net', 'target_policy_net'])
optimizers = namedtuple('optimizers', ['value_optimizer', 'policy_optimizer'])

value_net  = nets.ValueNetwork(
    state_dim, action_dim, hidden_dim).to(device)
policy_net = nets.PolicyNetwork(
    env, state_dim, action_dim, hidden_dim).to(device)

target_value_net  = nets.ValueNetwork(
    state_dim, action_dim, hidden_dim).to(device)
target_policy_net = nets.PolicyNetwork(
    env, state_dim, action_dim, hidden_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), 
                               value_net.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_policy_net.parameters(),
                               policy_net.parameters()):
    target_param.data.copy_(param.data)
    
ddpg_model = model(value_net=value_net, policy_net=policy_net, 
         target_value_net=target_value_net, 
         target_policy_net=target_policy_net) 
  
value_lr  = 1e-3
policy_lr = 1e-4
momentum_values = [0.01]
# optimal is 0.01
rewards_list = []
for momentum in momentum_values:
    print(f'Solving momentum = {momentum}')
    # policy_optimizer = torch.optim.SGD(policy_net.parameters(), 
    #                                    lr=policy_lr, momentum=momentum)
    value_optimizer  = optim.Adam(value_net.parameters(),  lr=value_lr)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
    
    ddpg_optimizer = optimizers(value_optimizer=value_optimizer, 
                  policy_optimizer=policy_optimizer)
    
    value_criterion = nn.MSELoss()
    
    replay_buffer_size = 1000000
    replay_buffer = ut.ReplayBuffer(replay_buffer_size)
    
    
    # max_epochs  = 15
    # max_steps   = 500
    max_epochs  = 25
    max_steps   = 300
    epoch   = 0
    rewards     = []
    batch_size  = 128
    sample_epochs = [0, int(max_epochs/2), max_epochs-1]
    sample_states = [np.array([i*0.1, env.constant_price]) for i in range(100)]
    sample_policies = []
    sample_policy_at_5  = []
    while epoch < max_epochs:
        state = env.reset()
        ou_noise.reset()
        episode_reward = 0
        # print(f'episode: ')
        for step in range(max_steps):
            print(f'\r episode: {epoch} '
                  f'reward {np.round(episode_reward/(step+1), 2)}', end='')
            action = policy_net.get_action(
                torch.FloatTensor(state).unsqueeze(0).to(device).float())
            action = ou_noise.get_action(action, step)
            next_state, reward = env.step(action)

            
            replay_buffer.push(state, action, reward, next_state, False)
            if len(replay_buffer) > batch_size:
                ddpg.update(ddpg_model, value_criterion, ddpg_optimizer, 
                            replay_buffer, batch_size)
            sample_state = torch.FloatTensor([5, env.constant_price]).to(device)
            sample_policy_at_5.append(policy_net.get_action(sample_state))
            
            
            state = next_state
            episode_reward += reward

        rewards.append(episode_reward/max_steps)
        if epoch in sample_epochs:
            sample_policies.append([policy_net.get_action(
                torch.tensor(s).to(device).float()) for s in sample_states])
        epoch += 1
    rewards_list.append(rewards.copy())
    print(f'reward {np.round(rewards[-1]/max_steps, 2)}                    ')
plt.figure(figsize=(10,5))
p_ind = 0    
for reward in rewards_list:
    plt.plot(reward, label=f'{momentum_values[p_ind]}')
    p_ind += 1
plt.legend() 
plt.grid()
plt.show()

# ut.plot(rewards, 'average return per epoch')
# plt.figure(figsize=(10,5))
# inventory_level = [state[0] for state in sample_states]
# plt.title('Evolving policy - price')
# for i in range(len(sample_epochs)):
#     plt.plot(inventory_level,
#              [sample_policies[i][j][0] for j in range(len(sample_states))], 
#              label=f'{sample_epochs[i]}')
# plt.legend() 
# plt.grid()
# plt.show() 

# plt.figure(figsize=(10,5))
# plt.title('Evolving policy - quantity')
# for i in range(len(sample_epochs)):
#     plt.plot(inventory_level,
#              [sample_policies[i][j][1] for j in range(len(sample_states))], 
#              label=f'{sample_epochs[i]}')
# plt.legend() 
# plt.grid()
# plt.show() 


plt.figure(figsize=(10,5))
plt.title('Policy at inventory= 5')
for i in range(len(sample_epochs)):
    plt.plot(sample_policy_at_5)
# plt.legend() 
plt.grid()
plt.show() 











