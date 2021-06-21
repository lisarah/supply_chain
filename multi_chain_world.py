# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 13:35:45 2021

@author: t-sarahli
"""
import numpy as np
from operator import itemgetter

import ddpg as ddpg

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

class World():
    def __init__(self, players, batch_size, mode = Distribution.first_come_first_serve):
        self.mode = mode
        self.players = players
        self.players_num = len(players)
        self.constant_seed_price = 1e-1
        self.batch_size = batch_size
        self.sample_policy  = [[] for p in players]
        
    def purchase(self, distributor, demand, supplier):
        available_quantity = supplier['state'][0]
        unfulfilled_demand = max(demand - available_quantity, 0)
        return unfulfilled_demand
    
    def step(self, step_ind):
        farmer_num  = self.players_num - 1 # level 1 players
        # Step 1: players decide on action
        for player in self.players:
            # update pstate
            player['state'] = player['chain'].state
            action = player['model'].policy_net.get_action(player['state']) 
            player['action'] = player['noise'].get_action(action, step_ind)
            # print(f'player {p_ind} action {players[p_ind]["action"]}')
            
        # Step 2: players link downstream and up stream actions
        # NOTE: only works currently for a two level horizontal chain with
        # one player on level 2 (the last player)
        # incoming price set by downstream player
        # print(f'constant seed price {self.constant_seed_price}')
        incoming_price = [self.constant_seed_price for p in range(farmer_num)]
        # distributor gets the average price as input
        average_price = sum([self.players[i]['action'][1] 
                             for i in range(farmer_num)]) / farmer_num
        # print (f'average price {average_price}')
        incoming_price.append(average_price)
        
        # quantity bought set by upstream player
        quantity_bought = [0 for i in range(farmer_num)]
        distributor = self.players[-1]
        # distributor gets market demand function
        quantity_bought.append(distributor['chain'].linear_demand(
            distributor['action'][0]))
        if self.mode is Distribution.equal: 
            for p_ind in range(farmer_num):# is a farmer
                demand = distributor['action'][1] / farmer_num
                unfulfilled = self.purchase(distributor, demand, player)
                quantity_bought[p_ind] = demand - unfulfilled
                if unfulfilled > 0: 
                    distributor['action'][1] += unfulfilled
        elif self.mode is Distribution.first_come_first_serve:
            # buy from cheapest player first
            player_offers = [p['action'][0] for p in self.players]
            unfulfilled =  distributor['action'][1]
            for tries in range(farmer_num): # all farmers try to satsify demand
                unfulfilled_prev = unfulfilled
                val, ind = min((val, ind) for (ind, val) in enumerate(player_offers))
                unfulfilled = self.purchase(distributor, unfulfilled, self.players[ind])
                quantity_bought[ind] = unfulfilled_prev - unfulfilled
                player_offers[ind] += 99999999
                
                if unfulfilled  == 0 : 
                    break
            if unfulfilled > 0: # after trying all farmers, still unfulfilled orders
                distributor['action'][1] += -unfulfilled
                
        # Step 3: players take a step            
        for player in self.players:        
            p_ind = player['ind']
            # print(f' incoming price {incoming_price[p_ind]}')
            # print(quantity_bought[p_ind])
            next_state, reward = player['chain'].step(
                player['action'], incoming_price[p_ind], quantity_bought[p_ind])

            player['replay_buffer'].push(player['state'], player['action'], 
                                         reward, next_state, False)
            if len(player['replay_buffer']) > self.batch_size:
                ddpg.update(player['model'], player['value_criterion'], 
                            player['optimizer'], player['replay_buffer'], 
                            self.batch_size)
            player['episode_reward'] += reward
            # update state and recordings    
            player['state'] = next_state
            self.sample_policy[player['ind']].append(
                player['model'].policy_net.get_action(player['state']))
            
