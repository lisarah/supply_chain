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
import multi_chain_world as world
import util as ut


model = namedtuple('model', ['value_net', 'policy_net', 
         'target_value_net', 'target_policy_net'])
optimizer = namedtuple('optimizers', ['value_optimizer', 'policy_optimizer'])
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

players = [
    {'chain': SupplyChain(3.5, 5)}, # farmer
    {'chain': SupplyChain(3.5, 5)}, # farmer
    # {'chain': SupplyChain(7, 5)}, # farmer
    {'chain': SupplyChain(7, 10)} # distributor
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
p_ind = 0

for player in players:
    player['ind'] = p_ind
    p_ind += 1
    player['model'] = model(**{
        'value_net': nets.ValueNetwork(
        state_dim, action_dim, hidden_dim).to(device),
        'policy_net': nets.PolicyNetwork(
        player['chain'], state_dim, action_dim, hidden_dim).to(device),
        'target_value_net': nets.ValueNetwork(
        state_dim, action_dim, hidden_dim).to(device),
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
max_epochs  = 100
max_steps   = 50
epoch  = 0
rewards     = [[] for p in players]
batch_size  = 128
# sample_epochs = [0, int(max_epochs/2), max_epochs-1]
# constant_price = 1e-1
# sample_states = [np.array([i*0.1, constant_price]) for i in range(100)]
# sample_policy  = [[] for p in players]
sample_inventory = 2.5

supply_world = world.World(players, batch_size)
# a_final = np.array([None, linear_demand(self, price)])
while epoch < max_epochs:
    for p in players:
        p['state'] = p['chain'].reset()
        p['noise'].reset()
        p['episode_reward'] = 0
    # print(f'episode: ')
    for step in range(max_steps):
        print(f'\r episode: {epoch} '
              f'r1 {np.round(players[0]["episode_reward"]/(step+1), 2)} ',
              f'r2 {np.round(players[1]["episode_reward"]/(step+1), 2)}', 
               f'r3 {np.round(players[2]["episode_reward"]/(step+1), 2)}',
              # f'r4 {np.round(players[3]["episode_reward"]/(step+1), 2)}', 
              end='')  
        supply_world.step(step)
        # # players decide on action
        # for p_ind in range(len(players)):
        #     # update pstate
        #     players[p_ind]['state'] = players[p_ind]['chain'].state
        #     action = players[p_ind]['model'].policy_net.get_action(
        #         players[p_ind]['state']) 
        #     players[p_ind]['action'] = players[p_ind]['noise'].get_action(action, step)
        #     # print(f'player {p_ind} action {players[p_ind]["action"]}')
        # for p_ind in range(len(players)):
        #     p_next = p_ind + 1
        #     if p_next < len(players):
        #         incoming_demand = players[len(players) - 1]['action'][1] / (len(players) - 1)
        #     else:
        #         incoming_demand = players[p_ind]['chain'].linear_demand(
        #             players[p_ind]['action'][0])
        #     # print(f' player {p_ind} faces demand {incoming_demand}')
        #     # a_prev = price set by downstream member
        #     a_prev = a_0 if p_ind != len(players) - 1 else sum([
        #         players[i]['action'] for i in range(len(players) - 1)]) / (len(players) - 1)
        #     # a_next = quantity set by upstream member
        #     a_next = np.array([
        #         None, 
        #         players[p_ind]['chain'].linear_demand(
        #             players[p_ind]['action'][0])])
        #     if p_ind + 1 < len(players):
        #         a_next[1] = players[len(players) - 1]['action'][1] / ((len(players) - 1))
        #         available_quantity = players[p_ind]['state'][0]
        #         if a_next[1] > available_quantity: 
        #             players[-1]['action'][1] += available_quantity - a_next[1]
        #             a_next[1] = available_quantity
                    
        #     next_state, reward = players[p_ind]['chain'].step(
        #         players[p_ind]['action'], a_prev, a_next)

        #     players[p_ind]['replay_buffer'].push(
        #         players[p_ind]['state'], players[p_ind]['action'], reward, 
        #         next_state, False)
        #     if len(players[p_ind]['replay_buffer']) > batch_size:
        #         ddpg.update(players[p_ind]['model'], 
        #                     players[p_ind]['value_criterion'], 
        #                     players[p_ind]['optimizer'], 
        #                     players[p_ind]['replay_buffer'], batch_size)
        #     players[p_ind]['state'] = next_state
        #     players[p_ind]['episode_reward'] += reward
        #     sample_policy[p_ind].append(
        #         players[p_ind]['model'].policy_net.get_action(
        #             players[p_ind]['chain'].state))
        #             # np.array([sample_inventory,  
        #             #           players[p_ind]['chain'].state[1]])))
            
    # print('') 
    for i in range(len(players)):
        rewards[i].append(players[i]['episode_reward'] / max_steps)
    epoch += 1

print('')
labels = ['farmer 1', 'farmer 2', 'distributor']
plt.figure(figsize=(10,5))
plt.title('average reward for each player')
[plt.plot(rewards[i], label = labels[i]) for i in range(len(players))]
plt.plot([rewards[0][i] + rewards[1][i] + rewards[2][i] for i in range(len(rewards[0]))], 
         linewidth=3, alpha=0.5, label='sum of all players ')
# plt.ylim([0,30])
plt.legend() 
plt.grid()
plt.show() 


plt.figure(figsize=(10,5))
plt.title('Price charged each iteration')
for i in range(len(players)):
    plt.plot([supply_world.sample_policy[i][j][0] 
              for j in range(len(supply_world.sample_policy[i]))], 
             label=labels[i])
plt.legend() 
plt.grid()
plt.show() 

plt.figure(figsize=(10,5))
plt.title('Quantity bought each iteration')
for i in range(len(players)):
    plt.plot([supply_world.sample_policy[i][j][1] 
              for j in range(len(supply_world.sample_policy[i]))], 
             label=labels[i])
plt.legend() 
plt.grid()
plt.show()