# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:39:58 2021

@author: Sarah Li
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


use_cuda =  torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    

class PolicyNetwork(nn.Module):
    def __init__(self, env, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(PolicyNetwork, self).__init__()
        
        self.env = env
        
        self.linear1 = nn.Linear(num_inputs, hidden_size).to(device)
        self.linear2 = nn.Linear(hidden_size, hidden_size).to(device)
        self.linear3 = nn.Linear(hidden_size, num_actions).to(device)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x)) * torch.Tensor(self.env.action_space.high).to(device)
        return x
    
    def get_action(self, state):
        state  = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]