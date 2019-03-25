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
import random
import matplotlib.pyplot as plt

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

# Hyper Parameters
BATCH_SIZE = 16
LR = 0.01  # learning rate
EPSILON = 0.9 # greedy policy
GAMMA = 0.9 # reward discount
TARGET_REPLACE_ITER = 20  # target update frequency
MEMORY_CAPACITY = 2000
# env = gym.make('CartPole-v0')
# env = env.unwrapped
# N_ACTIONS = env.action_space.n
# N_STATES = env.observation_space.shape[0]


env = RideHitch(filename='data/norm10000.txt')
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
        self.fc1 = nn.Linear(N_STATES, 100)
        nn.init.normal_(self.fc1.weight, 0, 0.1)
        # self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(100, 50)
        nn.init.normal_(self.fc2.weight, 0, 0.1)
        # self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, N_ACTIONS)
        nn.init.normal_(self.out.weight, 0, 0.1)
        # self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        s = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out(x)
        actions_value = F.relu(x)
        # for i in range(N_ACTIONS):
        #     if s[0][i * 6 + cap_idx - 1] <= 0:
        #         actions_value[0][i] = 0
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        print(self.eval_net)
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        if self.memory_counter > MEMORY_CAPACITY:
            ss = self.memory[:, :N_STATES]
            mean = np.mean(ss, axis=0)
            std = np.std(ss, axis=0)
        elif self.memory_counter > 0:
            ss = self.memory[:self.memory_counter, :N_STATES]
            mean = np.mean(ss, axis=0)
            std = np.std(ss, axis=0) + 1e-5
        else:
            mean = x
            std = 1
        x = torch.unsqueeze(torch.FloatTensor((x-mean)/std), 0)
        # input only one sample

        if np.random.uniform() < EPSILON:  # greedy
            actions_value = self.eval_net(x)
            # print('forward', self.eval_net(x)==self.eval_net.forward(x))
            # rule_actions_value = np.zeros(N_ACTIONS)
            # for i in range(N_ACTIONS):
            #     if x[0][i*6+cap_idx-1] > 0:
            #         rule_actions_value[i] = max(actions_value[0][i],0.001)
            #     else:
            #         rule_actions_value[i] = -1
            # # print(feasible_actions_num)
            # action = np.argmax(rule_actions_value)
            # no mask
            action = torch.max(actions_value, 1)[1].data.numpy()[0]

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
        #
        ss = self.memory[:, :N_STATES]
        mean = np.mean(ss, axis=0)
        std = np.std(ss, axis=0)
        # print('mean', mean)
        # print('std', std)
        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor((b_memory[:, :N_STATES]-mean)/std)
        # print('non-normalization', b_memory[:, :N_STATES])
        # print('after normalization', b_s)
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor((b_memory[:, -N_STATES:]-mean)/std)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        loss_record.append(loss.data.numpy())
        self.optimizer.step()

dqn = DQN()

print('\nCollecting experience...')
S_CAP = 5000
s_mem = np.zeros((S_CAP, N_STATES))
s_count = 0

for i_episode in range(100):
    start_time = time.time()
    s = env.reset(False)
    ep_r = 0
    matched = 0
    total_reward = 0
    # s_index = s_count % S_CAP
    # s_mem[s_index, :] = s
    while True:

        # if s_count > S_CAP:
        #     s_list = s_mem
        #     mean = np.mean(s_list, axis=0)
        #     std = np.std(s_list, axis=0)
        # elif s_count > 1:
        #     s_list = s_mem[:s_count, :]
        #     mean = np.mean(s_list, axis=0)
        #     std = np.std(s_list, axis=0) + 1e-5
        # else:
        #     mean = s
        #     std = 1
        # n_s = (s-mean)/std
        a = dqn.choose_action(s)
        # take action
        s_, r, done = env.step(a)
        # n_s_ = (s_-mean)/std
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
        # s_count += 1
        # s_index = s_count % S_CAP
        # s_mem[s_index, :] = s

    if s_count == 7000:
        print('s_mem', s_mem)
    if len(loss_record) > 2000:
        print('loss', np.mean(loss_record[-2000:]))
        # avg_los = np.mean(loss_record[-2000:])
    # for paras in dqn.eval_net.named_parameters():
    #     print(paras)
    print('Ep: ', i_episode,
          '| Ep_r: ', ep_r, '| Matched: ', matched, 'time', time.time() - start_time)

plt.plot(np.arange(len(loss_record)), loss_record)
plt.ylabel('Cost')
plt.xlabel('training steps')
plt.show()
# print(loss_record)
