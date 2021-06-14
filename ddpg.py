# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:39:58 2021

@author: Sarah Li
"""
import numpy as np

import torch


use_cuda =  torch.cuda.is_available()
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
            
            
            
def soft_q_update(model, criterion, optimizers, replay_buffer, batch_size, 
           gamma=0.99,
           mean_lambda=1e-3,
           std_lambda=1e-3,
           z_lambda=0.0,
           soft_tau=1e-2,
          ):
    state, action, reward, next_state, done = replay_buffer.sample(
        batch_size)

    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    expected_q_value = model.soft_q_net(state, action)
    expected_value   = model.value_net(state)
    new_action, log_prob, z, mean, log_std = model.policy_net.evaluate(state)


    target_value = model.target_value_net(next_state)
    next_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss = criterion.soft_q_criterion(expected_q_value, 
                                              next_q_value.detach())

    expected_new_q_value = model.soft_q_net(state, new_action)
    # print(log_prob.shape)
    # print(expected_new_q_value.shape)
    next_value = expected_new_q_value - log_prob
    # print(next_value.shape)
    # print(expected_value.shape)
    value_loss = criterion.value_criterion(expected_value, next_value.detach())

    log_prob_target = expected_new_q_value - expected_value
    policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
    

    mean_loss = mean_lambda * mean.pow(2).mean()
    std_loss  = std_lambda  * log_std.pow(2).mean()
    z_loss    = z_lambda    * z.pow(2).sum(1).mean()

    policy_loss += mean_loss + std_loss + z_loss

    optimizers.soft_q_optimizer.zero_grad()
    q_value_loss.backward()
    optimizers.soft_q_optimizer.step()

    optimizers.value_optimizer.zero_grad()
    value_loss.backward()
    optimizers.value_optimizer.step()

    optimizers.policy_optimizer.zero_grad()
    policy_loss.backward()
    optimizers.policy_optimizer.step()
    
    
    for target_param, param in zip(model.target_value_net.parameters(), 
                                   model.value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )            
            
            
            
            
            
            