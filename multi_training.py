# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:10:24 2021

Multi-agent RL training
@author: Sarah Li
"""
from comp_maddpg import MultiFarmerScenario
import models as nns

import torch.optim as optim
from collections import namedtuple


model = namedtuple('model', ['actor', 'critic', 'actor_targ', 'critic_targ'])
optimizer = namedtuple('optimizer', ['actor', 'critic'])
def get_trainers(world, obs_shapes, action_shapes, device, lr_a, lr_c):
    ac_nns = []
    optimizers = []
    for i in range(world.player_num()):
        ac_nns.append(model(**{
            'actor': 
                nns.actor(obs_shapes[i], action_shapes[i]).to(device), 
            'critic': 
                nns.critic(sum(obs_shapes), sum(action_shapes)).to(device),
            'actor_targ': 
                nns.actor(obs_shapes[i], action_shapes[i]).to(device),
            'critic_targ': 
                nns.critic(sum(obs_shapes), sum(action_shapes)).to(device)
            }))
        optimizers.append(optimizer(**{
            'actor': optim.Adam(ac_nns[-1].actor.parameters(), lr_a),
            'critic': optim.Adam(ac_nns[-1].critic.parameters(), lr_c)
            }))
        
def train(max_episode, num_farmers):
    #TODO: extend to other scenarios
    
    # step 1: Initiate multifarmer scenario
    world = MultiFarmerScenario.make_world(num_farmers)
    print('=============================')
    print('=1 MultiFarmerScenario is right ...')
    print('=============================')
    
    # step 2: Create agent neural networks and learning dynamics
    