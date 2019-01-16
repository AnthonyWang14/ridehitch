from env import RideHitch
from DQN import DeepQNetwork


def main():
    env = RideHitch(filename='data/text.txt')
    RL = DeepQNetwork(env.pool_size, env.state_num,
                        learning_rate=0.01,
                        reward_decay=0.9,
                        e_greedy=0.9,
                        replace_target_iter=200,
                        memory_size=2000,
                        output_graph=True)
    step = 0
    matched_list = []
    for episode in range(300):
        # init
        observation = env.reset(reset_seq=False)
        if episode % 100 == 0:
            print(episode)
        matched = 0
        while True:
            action = RL.choose_action(observation)
            observation_, reward, done = env.step(action)
            if reward > 0:
                matched += 1
            RL.store_transition(observation, action, reward, observation_)
            if (step > 200) and (step % 5 == 0):
                RL.learn()
            observation = observation_
            if done:
                break
            step += 1
        matched_list.append(matched)
    print(matched_list)
    RL.plot_cost()

if __name__ == '__main__':
    main()
    print('Finished!')
    