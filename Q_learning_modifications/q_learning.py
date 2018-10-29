import gym
import numpy as np
import random
from meleebot import MeleeBot


import gym
import numpy as np
import random

class Qlearning:
    def __init__(self, learning_rate, epsilon, environment):
        self.env = environment
        # if isinstance(self.env.observation_space,gym.spaces.box.Box):
        #     self.q_table = np.zeros([self.env.observation_space.shape[0], self.env.action_space.n])
        # else:
        #     self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        #self.q_table = np.zeros([env.observation_space.shape.n, env.action_space.n])
        self.q_table = np.zeros([5, 6, 5, 6, self.env.action_space.n])
        print("Shape of the Q-table:", self.q_table.shape)

        # Parameters for the Q alogrithm
        self.alpha = learning_rate
        self.epsilon = epsilon

        # Global variable for all methods
        self.gamma =  0.9
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay_rate = 0.01

        self.all_epochs = []

    def learn(self):
        state = self.env.reset()
        epochs, reward = 0, 0
        done = False

        while not done:
            if random.uniform(0,1) > self.epsilon:
                action = self.env.action_space.sample()
            else:
                action = np.argmax(self.q_table[state])

            next_state, reward, done, _ = self.env.step(action)
            next_state = next_state.astype(int).reshape(-1,1).tolist()  # finnes sikker en bedre måte, burde gjøres med state og i reseten
            next_max = np.max(self.q_table[next_state])

            self.q_table[state, action] = self.q_table[state, action] + self.alpha*(reward + self.gamma*next_max - self.q_table[state, action])

            state = next_state
            epochs += 1

            if reward != 0:
                print("Epoch:", epochs,"Reward: ",reward)

        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-0.1*self.epsilon)

    #def mapValues(self, value, min, max)


if __name__ == '__main__':
    bot = MeleeBot(iso_path="melee.iso")  # change to your path to melee v1.02 NTSC ISO
    print("Actions space size: ", bot.action_space.n)
    ql = Qlearning(0.1, 0.1, bot)
    bot.reset()
    while bot.CheckGameStatus == False:
        action = bot.action_space.sample()
        obv, reward, done, info = bot.step(action)
    ql.learn()
