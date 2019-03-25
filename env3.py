# state pool: the feasible supply for the latest request

import os, sys, random, time
import pandas as pd
import numpy as np
from utilities import *
import copy
import math


class RideHitch:

    def __init__(self, filename=None):

        random.seed(1)
        self.T_threshold = 60
        self.D_threshold = 70

        # self.T_threshold = 20
        # self.D_threshold = 20

        self.map_size = 100
        self.request_num = 1000
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
        self.state_num = 0
        self.state_pool_size = 100
        self.state_pool = []
        self.reset(reset_seq=True, filename=filename)
        self.rank_list = []
        self.state_rank_list = []
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
            t = bounded_normal(1 / 2 * self.time_max, 1 / 4 * self.time_max, 0, self.time_max)
            s_x = bounded_normal(1 / 2 * self.map_size, 1 / 4 * self.map_size, 0, self.map_size)
            s_y = bounded_normal(1 / 2 * self.map_size, 1 / 4 * self.map_size, 0, self.map_size)
            d_x = bounded_normal(1 / 2 * self.map_size, 1 / 4 * self.map_size, 0, self.map_size)
            d_y = bounded_normal(1 / 2 * self.map_size, 1 / 4 * self.map_size, 0, self.map_size)
            if request_type == 0:
                c = random.randint(self.supply_min, self.supply_max)
            else:
                c = random.randint(self.demand_min, self.demand_max)
            self.requests_list.append([request_type, t, s_x, s_y, d_x, d_y, c, i])
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
    def reset(self, reset_seq=True, filename=None):
        if reset_seq:
            if filename:
                self.generate_request_data(filename)
            else:
                self.generate_request_random()
        self.time_stamp = 0
        self.supply_pool = []
        self.rank_list = []
        self.latest_request = None
        while True:
            if self.time_stamp >= self.request_num:
                break
            self.latest_request = copy.deepcopy(self.requests_list[self.time_stamp])
            self.time_stamp += 1
            if self.latest_request[0] == 0:
                self.supply_pool.append(self.latest_request)
                self.rank_list.append(random.random())
            else:
                break
        return self.encode_state()

    def encode_state(self):
        self.state_pool = []
        self.state_rank_list = []
        for i, supply in enumerate(self.supply_pool):
            if check_match(supply, self.latest_request, self.T_threshold, self.D_threshold):
                self.state_pool.append(supply)
                self.state_rank_list.append(self.rank_list[i])
                if len(self.state_pool) > self.state_pool_size:
                    self.state_pool.pop(0)

        self.state_num = 6 * (self.state_pool_size + 1) + 1
        state = np.zeros(self.state_num)
        for i, supply in enumerate(self.state_pool):
            for j in range(6):
                state[6 * i + j] = supply[1 + j]
        for j in range(6):
            state[6 * self.state_pool_size + j] = self.latest_request[1 + j]
        # forget the time stamp
        state[-1] = 0
        return state

    # update environment
    # action format: the index of the chosen driver or -1 do nothing
    # return: states next, reward, if end
    def step(self, action):
        # get reward
        if action >= len(self.state_pool):
            reward = 0
        else:
            # print(len(self.supply_pool), self.supply_pool.index(self.state_pool[action]))
            reward = 1
            chosen_supply_idx = self.supply_pool.index(self.state_pool[action])
            self.supply_pool[chosen_supply_idx][cap_idx] -= self.latest_request[cap_idx]
        done = False

        while True:
            if self.time_stamp >= self.request_num:
                done = True
                break
            self.latest_request = copy.deepcopy(self.requests_list[self.time_stamp])
            self.time_stamp += 1
            if self.latest_request[type_idx] == 0:
                self.supply_pool.append(self.latest_request)
                self.rank_list.append(random.random())
            else:
                break
        # may need consider terminated state?
        s_next = self.encode_state()
        return s_next, reward, done


def greedy(action_for_choose, method, state_pool, state_rank_list):
    if method == "FIRST":
        action = action_for_choose[0]
    if method == "RANDOM":
        action = random.choice(action_for_choose)
    # TODO: add other method
    # if method == "MINCAP":
    #
    #     pass
    if method == "RANK":
        action = np.argmax(state_rank_list)
        # print(state_rank_list)
    return action


# baseline: greedy algorithm
if __name__ == '__main__':
    random.seed(1)
    env = RideHitch("data/taxi1000.txt")
    # env = RideHitch()
    # with open("data/norm1000.txt", "wt") as f:
    #     for req in env.requests_list:
    #         strarr = [str(item) for item in req]
    #         print(" ".join(strarr), file=f)
    for eps in range(10):
        s = env.reset(reset_seq=False)
        matched = 0
        # print(env.requests_list[0:10])
        print("seq size:", env.request_num, "state pool size:", env.state_pool_size)
        while True:
            action_for_choose = []
            demand = env.latest_request
            # use the state pool
            action_for_choose = range(len(env.state_pool))
            if len(action_for_choose) > 0:
                action = greedy(action_for_choose, 'FIRST', env.state_pool, env.state_rank_list)
                # action = action_for_choose[0]
            else:
                action = len(env.state_pool)
            s_, reward, done = env.step(action)
            if reward > 0:
                matched += 1
            if done:
                break
        # print(deg_list)
        print("eps", eps, "reward", matched)

