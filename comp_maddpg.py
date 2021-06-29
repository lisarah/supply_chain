# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:12:45 2021

dec-maddpg

@author: Sarah Li
"""
from multiagent.scenario import BaseScenario

from multi_chain_model import SupplyChain
import util as ut 


class Distribution:
    # equal: each farmer gets an equal share of the distributor's demand regardless of price
    equal = 1
    # fcfs: the farmer with the cheapest price offer gets the most distributor's demand
    # and if the distributor's demand is unmet after the cheapest farmer saturated 
    # his production capacity then it goes towards the next farmer
    first_come_first_serve = 2
    # probabilistic: each farmer gets a portion of the demand proportional to
    # (1/p_i)/ sum_j (1 / p_j) where p_i is the price charged by player i. 
    probabilistic = 3
    
def Player(object):
    def __init__(self, name=''):
        self.name = name
        self.chain = None
        self.action = None
        self.state = None
        self.action_callback = None
        self.action_noise = None
        self.reward = None
        
class World(object):
    def __init__(self):
        self.mode = Distribution.equal
        self.farmers = []
        self.distributors = []
        
        # communication channel dimensionality
        self.dim_c = 1
        # position/state dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        
    def reset(self):
        for p in self.farmers + self.distributors:
            p.state = p.chain.reset()
            p.noise.reset()
            
    # return all entities in the world
    @property 
    def entities(self):
        return self.farmers + self.distributors 
    
    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.farmers if agent.action_callback 
                is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.farmers if agent.action_callback 
                is not None]
    
    def player_num(self):
        #TODO: Extend to multi-level MADDPG
        return len(self.farmers + self.distributors)
    
    def rewards(self, player, player_key):
        """ player_key[0]: horizontal position, 
            player_key[1]: vertical position.
        """        
        #TODO: extend to more than 2 levels.
        if player_key[1] == 0:
            incoming_price = player.chain.constant_seed_price
        else:
            incoming_price = (
                sum(farmer.chain.action[1] for farmer in self.farmers) 
                              / len(self.farmers))
            
        #TODO: extend to more than 2 levels.
        if player_key[1] == 1:
            quantity_demanded = player.chain.linear_demand(player.action[0])
        else:
            #TODO: implement other distribution modes
            if self.mode is Distribution.equal:
                quantity_demanded = sum(distributor.action[1] for distributor 
                                        in self.distributors)
                quantity_demanded = quantity_demanded / len(self.farmers)
        
        # box constraint on player's own actions
        action = player.action
        cur_price = min(max(action[0], player.chain.action_space.low[0]), 
                        player.chain.action_space.high[0])
        quantity_bought = min(max(action[1], player.chain.action_space.low[1]), 
                              player.chain.action_space.high[1])
        
        quantity_sold = min(quantity_demanded, player.chain.state[0])
        reward = player.chain.reward_coeff * (
            quantity_sold * cur_price - quantity_bought * incoming_price)
        player.reward = reward
        
    def step(self, step_ind):
        """ This should be called before player update. """
        farmer_num = len(self.farmers)
        # Step 1: players decide on action
        for player in self.players:
            # update pstate
            player.state = player.chain.state
            action = player['model'].policy_net.get_action(player.states) 
            player.action = player.noise.get_action(action, step_ind)
            # print(f'player {p_ind} action {players[p_ind].action}')
        # Step 2: players link downstream and up stream actions
        # NOTE: only works currently for a two level horizontal chain with
        # one player on level 2 (the last player)
        # incoming price set by downstream player
        # print(f'constant seed price {self.constant_seed_price}')
        incoming_price = [self.constant_seed_price for p in range(farmer_num)]
        # distributor gets the average price as input
        average_price = sum([self.players[i].action[1] 
                             for i in range(farmer_num)]) / farmer_num
        # print (f'average price {average_price}')
        incoming_price.append(average_price)
        
        # quantity bought set by upstream player
        quantity_bought = [0 for i in range(farmer_num)]
        distributor = self.players[-1]
        # distributor gets market demand function
        quantity_bought.append(distributor.chain.linear_demand( 
            distributor.action[0]))
        if self.mode is Distribution.equal: 
            for p_ind in range(farmer_num):# is a farmer
                demand = distributor.action[1] / farmer_num
                unfulfilled = self.purchase(distributor, demand, player)
                quantity_bought[p_ind] = demand - unfulfilled
                if unfulfilled > 0: 
                    distributor.action[1] += unfulfilled
        elif self.mode is Distribution.first_come_first_serve:
            # buy from cheapest player first
            player_offers = [p.action[0] for p in self.players]
            unfulfilled =  distributor.action[1]
            for tries in range(farmer_num): # all farmers try to satsify demand
                unfulfilled_prev = unfulfilled
                val, ind = min((val, ind) for (ind, val) in enumerate(
                    player_offers))
                unfulfilled = self.purchase(distributor, unfulfilled, 
                                            self.players[ind])
                quantity_bought[ind] = unfulfilled_prev - unfulfilled
                player_offers[ind] += 99999999
                
                if unfulfilled  == 0 : 
                    break
            if unfulfilled > 0: # after trying all farmers, still unfulfilled orders
                distributor['action'][1] += -unfulfilled
                
    

class MultiFarmerScenario(BaseScenario):
    def make_world(self, num_farmers):
        world = World()
        world.num_players = num_farmers + 1
        # initialize farmers
        for p in range(num_farmers):
            farmer = Player(f'farmer_{p}')
            farmer.chain = SupplyChain(3.5, 5)
            farmer.action_noise = ut.OUNoise(farmer.chain.action_space)
            world.agents.append(farmer)
        # initialize distributor
        distributor = Player('distributor_0')
        distributor.chain = SupplyChain(7, 10)
        distributor.action_noise = ut.OUNoise(distributor.chain.action_space)

        
        # restart all players
        self.reset_world(world)
        
        return world
    
