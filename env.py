import os, sys, random, time
import pandas as pd
import numpy as np
from utilities import *
import copy

class RideHitch:

    def __init__(self,filename=None):
        
        random.seed(1)
        self.T_threshold = 50
        self.D_threshold = 50
        self.map_size = 100
        self.request_num = 10000
        self.requests_list = []


        # possible departure time window
        self.time_max = 144
        # some capacity parameters
        self.supply_min = 3
        self.supply_max = 6
        self.demand_min = 1
        self.demand_max = 6
        # if filename == None:
        #     self.generate_request_random()
        # else:
        #     self.generate_request_data(filename)

        self.time_stamp = 0
        self.supply_pool = []
        self.latest_request = None
        self.pool_size = 100
        self.state_num = 0
        self.reset()
        pass

    # generate all requests
    # request_type: 0-supply, 1-demand
    # t: departure time 
    # s: start point
    # d: destination
    # c: capacity

    def generate_request_random(self):
        self.requests_list = []
        for i in range(self.request_num):
            request_type = random.randint(0, 1)
            t = random.randint(0,self.time_max)
            s_x = random.randint(0,self.map_size)
            s_y = random.randint(0,self.map_size)
            d_x = random.randint(0,self.map_size)
            d_y = random.randint(0,self.map_size)
            if request_type == 0:
                c = random.randint(self.supply_min, self.supply_max)
            else:
                c = random.randint(self.demand_min, self.demand_max)
            self.requests_list.append([request_type,t,s_x,s_y,d_x,d_y,c])
        pass

    def generate_request_data(self, filename):
        self.requests_list = []

        with open(filename, 'rt') as f:
            for line in f:
                d = [int(i) for i in line.strip().split()]
                self.requests_list.append(d)
        self.request_num = len(self.requests_list)
        return

    # reset the environment
    def reset(self, reset_seq = True, filename=None):
        if reset_seq:
            if filename:
                self.generate_request_data(filename)
            else:
                self.generate_request_random()
        self.time_stamp = 0
        self.supply_pool = []
        self.latest_request = None
        while True:
            if self.time_stamp >= self.request_num:
                break
            self.latest_request = copy.deepcopy(self.requests_list[self.time_stamp])
            self.time_stamp += 1
            if self.latest_request[0] == 0:
                if len(self.supply_pool) >= self.pool_size:
                    self.supply_pool.pop(0)
                self.supply_pool.append(self.latest_request)
            else:
                break
        return self.encode_state_3()


    # def decode_request(self, req):
    #     t = req[1]
    #     s_x = req[2][0]
    #     s_y = req[2][1]
    #     d_x = req[3][0]
    #     d_y = req[3][1]
    #     c = req[4]
    #     return t, s_x, s_y, d_x, d_y, c

    def encode_state(self):
        self.state_num = self.pool_size*12+self.request_num
        state = np.zeros(self.state_num)
        # encode supply and demand
        for i, supply in enumerate(self.supply_pool):
            for j in range(6):
                state[12*i+j] = supply[1+j]
                state[12*i+j+6] = self.latest_request[1+j]
        # encode time stamp
        state[self.pool_size*12+self.time_stamp-1] = 1
        return state

    # one_hot encoding for the map
    # 2*map_size*4*self.time_max+self.request_num (1 means the time_stamp) 2 means the supply and demand
    # one_hot encoding has no order, the action space need to reconsider
    def encode_state_2(self):
        self.state_num = 2*self.map_size*4*self.time_max+self.request_num
        self.state = np.zeros(self.state_num)

        # encode supply
        for supply in self.supply_pool:
            t, s_x, s_y, d_x, d_y, c = self.decode_request(supply)
            self.state[t*self.map_size*4+s_x] = c
            self.state[t*self.map_size*4+self.map_size+s_y] = c
            self.state[t*self.map_size*4+self.map_size*2+d_x] = c
            self.state[t*self.map_size*4+self.map_size*3+d_y] = c

        # encode demand
        t, s_x, s_y, d_x, d_y, c = self.decode_request(self.latest_request)
        self.state[(t+self.time_max)*self.map_size*4+s_x] = c
        self.state[(t+self.time_max)*self.map_size*4+self.map_size+s_y] = c
        self.state[(t+self.time_max)*self.map_size*4+self.map_size*2+d_x] = c
        self.state[(t+self.time_max)*self.map_size*4+self.map_size*3+d_y] = c
        # encode time_stamp

        self.state[2*self.map_size*4*self.time_max+self.time_stamp] = 1
        return self.state
    
    # decode one_hot state
    def decode_state_2(self):
        self.supply_pool_no_order = []

    def encode_state_3(self):
        self.state_num = 6*(self.pool_size+1) + 1
        state = np.zeros(self.state_num)
        for i, supply in enumerate(self.supply_pool):
            for j in range(6):
                state[6*i+j] = supply[1+j]
        for j in range(6):
            state[6*self.pool_size+j] = self.latest_request[1+j]
        state[-1] = self.time_stamp
        return state


    # check match
    def check_match(self, supply, demand):
        if supply[-1] < demand[-1]:
            return False
        if np.abs(supply[1]-demand[1]) > self.T_threshold:
            return False
        a = [supply[2], supply[3]]
        b = [supply[4], supply[5]]
        c = [demand[2], demand[3]]
        d = [demand[4], demand[5]]
        old_path = dist(a,b)
        new_path = dist(a,c) + dist(c,d) + dist(d,b)
        detour = new_path - old_path
        if detour > self.D_threshold:
            return False
        return True


    # update environment
    # action format: the index of the chosen driver or -1 do nothing
    # return: states next, reward, if end
    def step(self, action):
        # get reward
        if action >= len(self.supply_pool):
            reward = 0
        else:
            if self.check_match(self.supply_pool[action], self.latest_request):
                self.supply_pool[action][-1] -= self.latest_request[-1]
                reward = 1
            else:
                reward = -1
        done = False
        while True:
            if self.time_stamp >= self.request_num:
                done = True
                break
            self.latest_request = copy.deepcopy(self.requests_list[self.time_stamp])
            self.time_stamp += 1
            if self.latest_request[0] == 0:
                if len(self.supply_pool) >= self.pool_size:
                    self.supply_pool.pop(0)
                self.supply_pool.append(self.latest_request)
            else:
                break
        # may need consider terminated state?
        s_next = self.encode_state_3()
        return s_next, reward, done

# baseline: greedy algorithm
if __name__ == '__main__':
    random.seed(1)
    env = RideHitch()
    with open("data/test10000.txt", "wt") as f:
        for req in env.requests_list:
            strarr = [str(item) for item in req]
            print(" ".join(strarr), file=f)
    # pass 
    for eps in range(5):
        s = env.reset(reset_seq = True, filename="data/test10000.txt")
        matched = 0
        # print(env.requests_list[0:10])
        while True:
            action_for_choose = []
            demand = env.latest_request  
            s = env.encode_state()
            for i in range(env.pool_size):
                supply_chosen = [0,0,0,0,0,0,0]
                for j in range(6):
                    supply_chosen[1+j] = s[12*i+j]
                if env.check_match(supply_chosen, demand):
                    action_for_choose.append(i)
            # for i in range(len(env.supply_pool)):
            #     if env.check_match(env.supply_pool[i], demand):
            #         action_for_choose.append(i)
            if len(action_for_choose) > 0:
                # print(len(action_for_choose))
                action = random.choice(action_for_choose)
            else:
                action = 0
            s_, reward, done = env.step(action)
            if reward > 0:
                matched += 1
            if done:
                break
        print("eps", eps, "reward", matched)
