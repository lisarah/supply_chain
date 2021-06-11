# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 17:12:25 2021

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
from multi_chain_model import SupplyChain
import util as ut


model = namedtuple('model', ['value_net', 'policy_net', 
         'target_value_net', 'target_policy_net'])
optimizer = namedtuple('optimizers', ['value_optimizer', 'policy_optimizer'])
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

players = [
    {'chain': SupplyChain()},
    {'chain': SupplyChain()}
    ]
p_num = len(players)
# all players have identical state-action spaces
state_dim  = players[0]['chain'].observation_space.shape[0]
action_dim = players[0]['chain'].action_space.low.shape[0]

hidden_dim = 256


value_lr  = 1e-3
policy_lr = 1e-4
policy_moment = 1e-2
replay_buffer_size = 1000000

for player in players:
    player['model'] = model(**{
        'value_net': nets.ValueNetwork(
        state_dim * p_num, action_dim * p_num, hidden_dim).to(device),
        'policy_net': nets.PolicyNetwork(
        player['chain'], state_dim, action_dim, hidden_dim).to(device),
        'target_value_net': nets.ValueNetwork(
        state_dim * p_num, action_dim * p_num, hidden_dim).to(device),
        'target_policy_net':  nets.PolicyNetwork(
        player['chain'], state_dim, action_dim, hidden_dim).to(device)
        }) # ddpg_model
    
    for target_param, param in zip(player['model'].target_value_net.parameters(), 
                                   player['model'].value_net.parameters()):
        target_param.data.copy_(param.data)
    
    for target_param, param in zip(player['model'].target_policy_net.parameters(),
                                   player['model'].policy_net.parameters()):
        target_param.data.copy_(param.data)
    
    player['optimizer'] = optimizer(**{
        'value_optimizer': optim.Adam(player['model'].value_net.parameters(),  
                                      lr=value_lr),
        'policy_optimizer': torch.optim.SGD(
            player['model'].policy_net.parameters(), lr=policy_lr, 
            momentum=policy_moment)
        }) # ddpg_optimizer
    player['value_criterion'] = nn.MSELoss()
    player['replay_buffer'] = ut.ReplayBuffer(replay_buffer_size)
    player['noise'] = ut.OUNoise(player['chain'].action_space)
    player['state'] = None
    player['action'] = None
    player['episode_reward'] = 0

# simultaneous training, different objectives
# max_epochs  = 15
# max_steps   = 500
max_epochs  = 50
max_steps   = 200
epoch   = 0
rewards     = [[] for p in players]
batch_size  = 128
sample_epochs = [0, int(max_epochs/2), max_epochs-1]
constant_price = 2.5
sample_states = [torch.FloatTensor([i*0.1, constant_price]).to(device) 
                 for i in range(100)]
sample_policy  = [[] for p in players]
sample_inventory = 2.5
a_0= np.array([1e-1, None])
# a_final = np.array([None, linear_demand(self, price)])
while epoch < max_epochs:
    for p in players:
        p['state'] = p['chain'].reset()
        p['noise'].reset()
        p['episode_reward'] = 0
    # print(f'episode: ')
    for step in range(max_steps):
        print(f'\r episode: {epoch} '
              f'r1 {np.round(players[0]["episode_reward"]/(step+1), 2)}  ',
               f'r2 {np.round(players[1]["episode_reward"]/(step+1), 2)}  ', 
              end='')  
        
        for p_ind in range(len(players)):
            # update pstate
            players[p_ind]['state'] = players[p_ind]['chain'].state
            # find the action at current state    
            action = players[p_ind]['model'].policy_net.get_action(
                torch.FloatTensor(players[p_ind]['state']).to(device)) 
            players[p_ind]['action'] = players[p_ind]['noise'].get_action(action, step)
            
        for p_ind in range(len(players)):
            
            p_next = p_ind + 1
            if p_next < len(players):
                incoming_demand = players[p_next]['action'][1]
            else:

                incoming_demand = players[p_ind]['chain'].linear_demand(
                    players[p_ind]['action'][0])
            a_prev = a_0 if p_ind == 0 else players[p_ind-1]['action']
            a_next = np.array([
                None, 
                players[p_ind]['chain'].linear_demand(
                    players[p_ind]['action'][0])])
            if p_ind + 1 < len(players):
                a_next = players[p_ind+1]['action']
            next_state, reward = players[p_ind]['chain'].step(
                players[p_ind]['action'], a_prev, a_next)

            p['replay_buffer'].push(p['state'], p['action'], reward, 
                                    next_state, False)
            if len(p['replay_buffer']) > batch_size:
                ddpg.update(p['model'], p['value_criterion'], p['optimizer'], 
                        p['replay_buffer'], batch_size)
            p['state'] = next_state
            p['episode_reward'] += reward
            sample_state = torch.FloatTensor([sample_inventory, 
                                              p['chain'].state[1]]).to(device)
            sample_policy[p_ind].append(
                p['model'].policy_net.get_action(sample_state))
            
    # print('') 
    for i in range(len(players)):
        rewards[i].append(players[i]['episode_reward']/max_steps)
    epoch += 1

print('')
plt.figure(figsize=(10,5))
plt.title('average reward')
[plt.plot(rewards[i], label = f'{i}') for i in range(len(players))]
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
plt.title(f'Price at inventory= {sample_inventory}')
for i in range(len(sample_policy)):
    plt.plot([sample_policy[j][0] #[i][j][0]
              for j in range(len(sample_policy[i]))], label=f'{i} price')
plt.legend() 
plt.grid()
plt.show() 

plt.figure(figsize=(10,5))
plt.title(f'Quantity at inventory= {sample_inventory}')
for i in range(len(sample_policy)):
    plt.plot([sample_policy[i][j][1] 
              for j in range(len(sample_policy[i]))], label=f'{i} quantity')
plt.legend() 
plt.grid()
plt.show()