# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:59:07 2021

@author: Sarah Li
"""
from collections import namedtuple

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
# ou_noise = ut.OUNoise(env.action_space)

# state_dim  = env.observation_space.shape[0]
state_dim = 1
# action_dim = env.action_space.shape[0]
action_dim = 2
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

value_optimizer  = optim.Adam(value_net.parameters(),  lr=value_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

ddpg_optimizer = optimizers(value_optimizer=value_optimizer, 
              policy_optimizer=policy_optimizer)

value_criterion = nn.MSELoss()

replay_buffer_size = 1000000
replay_buffer = ut.ReplayBuffer(replay_buffer_size)


max_frames  = 12000
max_steps   = 500
frame_idx   = 0
rewards     = []
batch_size  = 128


while frame_idx < max_frames:
    state = env.reset()
#     ou_noise.reset()
    episode_reward = 0
    if frame_idx %100 == 0:
        print(f'\r  step{frame_idx}', end='')
    for step in range(max_steps):
        
        action = policy_net.get_action(state)
#         action = ou_noise.get_action(action, step)
        next_state, reward = env.step(action)
        
        replay_buffer.push(state, action, reward, next_state, False)
        if len(replay_buffer) > batch_size:
            ddpg.ddpg_update(ddpg_model, value_criterion, ddpg_optimizer, 
                             replay_buffer, batch_size)
        
        state = next_state
        episode_reward += reward
        frame_idx += 1
    
    rewards.append(episode_reward)


ut.plot(frame_idx, rewards)














