# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 16:47:51 2021

Copied from MADDPG_torch-master
@author: Sarah Li
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class abstract_agent(nn.Module):
    def __init__(self):
        super(abstract_agent, self).__init__()
    
    def act(self, input):
        policy, value = self.forward(input) # flow the input through the nn
        return policy, value

class critic(abstract_agent):
    def __init__(self, obs_shape_n, action_shape_n, hidden_num=64):
        super(critic, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(action_shape_n+obs_shape_n, hidden_num)
        self.linear_c2 = nn.Linear(hidden_num, hidden_num)
        self.linear_c = nn.Linear(hidden_num, 1)

        self.reset_parameters()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=gain)
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=gain)
        nn.init.xavier_uniform_(self.linear_c.weight, gain=gain)

    def forward(self, obs_input, action_input):
        """
        input_g: input_global, input features of all agents
        """
        x_cat = self.LReLU(self.linear_c1(torch.cat([obs_input, action_input], dim=1)))
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value

class actor(abstract_agent):
    def __init__(self, num_inputs, action_size, hidden_num=64):
        super(actor, self).__init__()
        self.tanh= nn.Tanh()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_a1 = nn.Linear(num_inputs, hidden_num)
        self.linear_a2 = nn.Linear(hidden_num, hidden_num)
        self.linear_a = nn.Linear(hidden_num, action_size)

        self.reset_parameters()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.linear_a1.weight, gain=gain)
        nn.init.xavier_uniform_(self.linear_a2.weight, gain=gain)
        nn.init.xavier_uniform_(self.linear_a.weight, gain=gain)
    
    def forward(self, input, model_original_out=False):
        """
        The forward func defines how the data flows through the graph(layers)
        flag: 0 sigle input 1 batch input
        """
        x = self.LReLU(self.linear_a1(input))
        x = self.LReLU(self.linear_a2(x))
        model_out = self.linear_a(x)
        u = torch.rand_like(model_out)
        policy = F.softmax(model_out - torch.log(-torch.log(u)), dim=-1)
        if model_original_out == True:   return model_out, policy # for model_out criterion
        return policy
