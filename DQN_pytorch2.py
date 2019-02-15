"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/

Dependencies:
torch: 0.4
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities import *
from env2 import RideHitch


# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01  # learning rate
EPSILON = 0.99  # greedy policy
GAMMA = 1  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 2000
# env = gym.make('CartPole-v0')
# env = env.unwrapped
# N_ACTIONS = env.action_space.n
# N_STATES = env.observation_space.shape[0]


env = RideHitch(filename='data/norm1000.txt')
N_ACTIONS = env.pool_size
N_STATES = env.state_num
T_threshold = env.T_threshold
D_threshold = env.D_threshold
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape
# to confirm the shape


loss_record = []


class Net(nn.Module):

    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 300)
        # self.fc1.weight.data.normal_(0, 0.1)  # initialization
        nn.init.xavier_normal_(self.fc1.weight.data)
        self.fc2 = nn.Linear(300, 100)
        # self.fc2.weight.data.normal_(0, 0.1)
        nn.init.xavier_normal_(self.fc2.weight.data)
        self.fc3 = nn.Linear(100, N_ACTIONS)
        # self.fc3.weight.data.normal_(0, 0.1)  # initialization
        nn.init.xavier_normal_(self.fc3.weight.data)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        actions_value = F.relu(x)
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
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:  # greedy
            actions_value = self.eval_net.forward(x)
            rule_actions_value = np.zeros(N_ACTIONS)

            for i in range(N_ACTIONS):
                if x[0][i] > 0:
                    rule_actions_value[i] = actions_value[0][i]
                else:
                    rule_actions_value[i] = -999999
            # print(rule_actions_value)
            action = np.argmax(rule_actions_value)
            # action = torch.max(actions_value, 1)[1].data.numpy()
            # # action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
            # action = action[0]
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

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        loss_record.append(loss.data.numpy())
        # print(loss.data.numpy())
        self.optimizer.step()


dqn = DQN()

print('\nCollecting experience...')
for i_episode in range(100):
    s = env.reset(False)
    ep_r = 0
    matched = 0
    while True:
        a = dqn.choose_action(s)
        # take action
        s_, r, done = env.step(a)
        dqn.store_transition(s, a, r, s_)
        ep_r += r
        if r > 0:
            matched += 1
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            # if done:
            #     print('Ep: ', i_episode,
            #           '| Ep_r: ', ep_r, '| Matched: ', matched)

        if done:
            break
        s = s_
    print('Ep: ', i_episode,
          '| Ep_r: ', ep_r, '| Matched: ', matched)
# print(loss_record)
