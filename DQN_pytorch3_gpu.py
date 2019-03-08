"""
This DQN-pytorch for env_weighted (encode the feasible driver as state)

Dependencies:
torch: 0.4
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities import *
from env_weighted import RideHitch
import time

# Hyper Parameters
BATCH_SIZE = 16
LR = 0.01  # learning rate
EPSILON = 1  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 20  # target update frequency
MEMORY_CAPACITY = 2000
# env = gym.make('CartPole-v0')
# env = env.unwrapped
# N_ACTIONS = env.action_space.n
# N_STATES = env.observation_space.shape[0]


env = RideHitch(filename='taxi2k/0')
N_ACTIONS = env.state_pool_size
N_STATES = env.state_num
T_threshold = env.T_threshold
D_threshold = env.D_threshold
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape
# to confirm the shape


loss_record = []


class Net(nn.Module):

    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 500)
        nn.init.xavier_normal_(self.fc1.weight)
        self.out = nn.Linear(500, N_ACTIONS)
        nn.init.xavier_normal_(self.out.weight)
        # self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = self.out(x)
        actions_value = F.relu(x)
        return actions_value

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        # use cuda
        self.eval_net.cuda()
        self.target_net.cuda()

        print(self.eval_net)
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:  # greedy
            xcuda = x.cuda()
            actions_value = self.eval_net.forward(xcuda)
            rule_actions_value = np.zeros(N_ACTIONS)
            for i in range(N_ACTIONS):
                if x[0][i*6+cap_idx-1] > 0:
                    rule_actions_value[i] = max(actions_value[0][i],0.001)
                else:
                    rule_actions_value[i] = -1
            # print(feasible_actions_num)
            action = np.argmax(rule_actions_value)
        else:  # random
            action = np.random.randint(0, N_ACTIONS)
            # action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        ss = self.memory[:, :N_STATES]
        mean = np.mean(ss, axis=0)
        std = np.nanstd(ss, axis=0)+1e-3

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor((b_memory[:, :N_STATES]-mean)/std).cuda()
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int)).cuda()
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2]).cuda()
        b_s_ = torch.FloatTensor((b_memory[:, -N_STATES:]-mean)/std).cuda()

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        loss_record.append(loss.data.cpu())
        # print(loss.data.numpy())
        self.optimizer.step()



dqn = DQN()

print('\nCollecting experience...')
for i_episode in range(30):
    start_time = time.time()
    s = env.reset(False)
    ep_r = 0
    matched = 0
    total_reward = 0
    while True:
        a = dqn.choose_action(s)
        # take action
        s_, r, done = env.step(a)
        dqn.store_transition(s, a, r, s_)
        ep_r += r
        if r > 0:
            matched += 1
            total_reward += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            # if done:
            #     print('Ep: ', i_episode,
            #           '| Ep_r: ', ep_r, '| Matched: ', matched)

        if done:
            break
        s = s_
    print('Ep: ', i_episode,
          '| Ep_r: ', ep_r, '| Matched: ', matched, 'time', time.time()-start_time)
print(dqn)

import matplotlib.pyplot as plt
plt.plot(np.arange(len(loss_record)), loss_record)
plt.ylabel('Cost')
plt.xlabel('training steps')
plt.show()
# print(loss_record)
