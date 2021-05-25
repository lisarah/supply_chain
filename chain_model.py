# -*- coding: utf-8 -*-
"""
Created on Mon May 24 18:26:55 2021

@author: craba
"""
import numpy as np
from collections import namedtuple


space = namedtuple('space', ['low', 'high'])

class SupplyChain():
    def __init__(self, T=100):
        """ Obeservation space:
            - curent inventory - Q_c
            
            Exogenous signals:
            - cost per item - P_in
            - forecast_demand - D_1
            - forecast_demand - D_2
            Wrap this into the transition itself - generates stochasticity.
        
            
            Demand function:
                Q_D = forecast_demand_1 - a_demand * P_out + epsilon
                a_demand = coefficient
                epsilon = Gaussian variable, mean = d, variance = sigma
                
            Actions:
                Price = nonnegative price of outgoing products.
                Quantity  = nonnegative quantity of products bought.
            
            Holding cost: 
                holding cost = a_hold * inventory
            Demand unsatisfaction cost: 
                cost = a_unsat * quantity_deficit
        """
        self.demand_mean = 1
        self.demand_sigma = 0.1
        self.a_demand = 2e-1
        self.c_demand = 1
        self.a_price = 5e-1
        self.c_price = 1e0
        self.constant_price = 1e-1
        self.a_hold = 5e-2
        self.a_unsatisfied = 5e-2
        self.reward_coeff = 1e1

        
        # get rid of forecast_demand
        self.observation_space = np.zeros((1))
        # price is at most 5, quantity is at most 2.
        self.action_space = space(low = np.array([0,0]), 
                                  high = np.array([5,2]))
        self.state=np.zeros((self.observation_space.shape[0]))
        

        self.reset()
    
    def get_random_demand(self, price):
        Q = self.state[2] #+ np.random.normal(0, self.demand_sigma)
        # price cannot be negative.
        return max(Q - self.a_demand * max(price, 0), 0)
    
    def linear_demand(self, price):
        demand = max(self.c_demand - self.a_demand * price , 0)
        demand += np.random.normal(0, self.demand_sigma) # add noise
        return demand
    
    def linear_price(self, quantity):
        return max(self.a_price - self.a_price * quantity )
    
    def holding_cost(self, quantity):
        return self.a_hold * quantity
    
    def demand_unsatisfaction(self, quantity_sold, cur_demand):
        return self.a_unsatisfied * max(cur_demand - quantity_sold, 0)
    
    def reset(self):
        self.state[0] = np.random.uniform(0, 10)# current inventory
        # self.state[1] = 1e-1 # cost per item bought
        # self.state[2] = max(np.random.normal(self.demand_mean,self.demand_sigma), 0) # demand_forecast #1
        # self.state[3] = max(np.random.normal(self.demand_mean,self.demand_sigma), 0) # demand_forecast #2
        
        return self.state.copy()
        
    def update_state(self, quantity_bought, quantity_sold):
        self.state[0] += - quantity_sold + quantity_bought
        self.state[0] = min(10, max(self.state[0], 0))
        # cost per item stays constant for now
        # self.state[2] = 1* self.state[3]
        # self.state[3] = max(np.random.normal(self.demand_mean,
        #                                      self.demand_sigma), 0) 
        
    def step(self, action, debug=False):
        # print(f'action taken {action}')
        cur_price = min(max(action[0], self.action_space.low[0]), 
                        self.action_space.high[0])
        quantity_bought = min(max(action[1], self.action_space.low[1]), 
                              self.action_space.high[1])
        if debug:
            print(f'cur_price {cur_price}')
            print(f'quantity_bought {quantity_bought}')
        demand_out = self.linear_demand(cur_price)
        quantity_sold  = min(demand_out, self.state[0])
        reward = self.reward_coeff *(quantity_sold * cur_price 
                  - quantity_bought * self.constant_price)
        holding_cost = self.holding_cost(
            quantity_bought + self.state[0] - quantity_sold)
        demand_penalty = self.demand_unsatisfaction(quantity_sold, demand_out)
       
        reward += -holding_cost
        reward += -demand_penalty
        
        self.update_state(quantity_bought, quantity_sold)
        
        return self.state, reward
        
        
        