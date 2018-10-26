import gym
import numpy as np
import random
from meleebot import MeleeBot


class Qlearning:
    def __init__(self, learning_rate, epsilon):
        self.q_table = np.zeros([5 * 5 * 20, bot.action_space.n])

        # Parameters for the Q alogrithm
        self.gamma = 0.9
        self.alpha = learning_rate  # learning rate

        self.epsilon = epsilon
        self.max_epsilon = 1
        self.min_epsilon = 0.01
        self.decay_rate = 0.01

        self.all_epochs = []
        self.all_penalities = []

    def learn(self):
        for i in range(1, 100):
            state, state2 = bot.reset()
            state = state.astype(int)
            state2 = state2.astype(int)

            epochs, penalties, reward = 0, 0, 0
            done = False
            while not done:
                if random.uniform(0, 1) < self.epsilon:
                    action = bot.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])

                if random.uniform(0, 1) < self.epsilon:
                    action2 = bot.action_space.sample()
                else:
                    action2 = np.argmax(self.q_table[state2])

                (next_state1, next_state2), (reward1, reward2), done, info = bot.step(action, action2)
                next_state1 = next_state1.astype(int)
                next_state2 = next_state2.astype(int)
                next_max = np.argmax(self.q_table[next_state1])
                next_max2 = np.argmax(self.q_table[next_state2])

                self.q_table[next_state1, action] = self.q_table[next_state1, action] + self.alpha * \
                                                    (reward1 + self.gamma * next_max) - self.q_table[
                                                        next_state1, action]
                self.q_table[next_state2, action2] = self.q_table[next_state2, action2] + self.alpha * \
                                                    (reward2 + self.gamma * next_max2) - self.q_table[
                                                        next_state2, action2]

                state = next_state1
                state2 = next_state2
                epochs += 1

            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-0.1 * self.epsilon)
            if i % 10 == 0:
                print("Episode: {}".format(i))


bot = MeleeBot(iso_path="melee.iso", player_control=False)  # change to your path to melee v1.02 NTSC ISO
ql = Qlearning(0.1, 0.1)
ql.learn()
