# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 13:46:53 2021

@author: Sarah Li
"""
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import ddpg as ddpg
import soft_actor_critic as nets
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
         'target_value_net', 'soft_q_net'])
optimizers = namedtuple('optimizers', ['value_optimizer', 'policy_optimizer', 
                                       'soft_q_optimizer'])
criterions = namedtuple('criterions', ['value_criterion', 'soft_q_criterion'])

value_net        = nets.ValueNetwork(state_dim, hidden_dim).to(device)
target_value_net = nets.ValueNetwork(state_dim, hidden_dim).to(device)

soft_q_net = nets.SoftQNetwork(env, state_dim, action_dim, hidden_dim).to(device)
policy_net = nets.PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), 
                               value_net.parameters()):
    target_param.data.copy_(param.data)
    
train_model = model(soft_q_net=soft_q_net, value_net=value_net, 
                    policy_net=policy_net, target_value_net=target_value_net) 

train_criterion = criterions(value_criterion=nn.MSELoss(), 
                             soft_q_criterion=nn.MSELoss())
value_lr  = 3e-4
soft_q_lr = 3e-4
policy_lr = 3e-4
train_optimizers = optimizers(
    value_optimizer=optim.Adam(value_net.parameters(), lr=value_lr),
    policy_optimizer=optim.Adam(policy_net.parameters(), lr=policy_lr),
    soft_q_optimizer=optim.Adam(soft_q_net.parameters(), lr=soft_q_lr))
    


replay_buffer_size = 1000000
replay_buffer = ut.ReplayBuffer(replay_buffer_size)

def plot(frame_idx, rewards):
    plt.figure(figsize=(8,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()
    
max_frames  = 40000
max_steps   = 500
frame_idx   = 0
rewards     = []
batch_size  = 128

while frame_idx < max_frames:
    state = env.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        print(f'\r frame: {frame_idx} '
              f'reward {np.round(episode_reward/(step+1), 2)}', end='')
        action = policy_net.get_action(state)
        next_state, reward, = env.step(action)
        
        replay_buffer.push(state, action, reward, next_state, False)
        if len(replay_buffer) > batch_size:
            ddpg.soft_q_update(train_model, train_criterion, train_optimizers, 
                               replay_buffer, batch_size)

        state = next_state
        episode_reward += reward
        frame_idx += 1
        
        
    rewards.append(episode_reward)
plot(frame_idx, rewards)    
    
    
    