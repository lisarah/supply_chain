# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:39:58 2021

@author: Sarah Li
"""
import numpy as np

import torch



use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

def update(model, criterion, optimizers, replay_buffer, batch_size, 
           gamma = 0.99, min_value=-np.inf, max_value=np.inf, soft_tau=1e-2):
    
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    
    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    policy_loss = model.value_net(state, model.policy_net(state))
    policy_loss = -policy_loss.mean()

    next_action    = model.target_policy_net(next_state)
    target_value   = model.target_value_net(next_state, next_action.detach())
    expected_value = reward + (1.0 - done) * gamma * target_value
    expected_value = torch.clamp(expected_value, min_value, max_value)

    value = model.value_net(state, action)
    value_loss = criterion(value, expected_value.detach())


    optimizers.policy_optimizer.zero_grad()
    policy_loss.backward()
    optimizers.policy_optimizer.step()

    optimizers.value_optimizer.zero_grad()
    value_loss.backward()
    optimizers.value_optimizer.step()

    for target_param, param in zip(model.target_value_net.parameters(), 
                                   model.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    for target_param, param in zip(model.target_policy_net.parameters(), 
                                   model.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
            
            
            
            
            
            
            
            
            
            