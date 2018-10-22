import gym
import numpy as np
import random
from meleebot import MeleeBot


class Qlearning:
    def __init__(self, learning_rate, epsilon):
        self.q_table = np.zeros([bot.observation_space.shape[0], bot.action_space.n])

        # Parameters for the Q alogrithm
        self.gamma =  0.9
        self.alpha = learning_rate     #learning rate

        self.epsilon = epsilon
        self.max_epsilon = 1
        self.min_epsilon = 0.01
        self.decay_rate = 0.01

        self.all_epochs = []
        self.all_penalities = []
        self.penalities = 0


    def learn(self):
        for i in range(1, 100001):
            self.state = bot.reset()

            self.epochs, self.penalities, self.reward = 0, 0, 0
            done = False
            while not done:
                if random.uniform(0,1) < self.epsilon:
                    action = bot.action_space.sample()
                else:
                    action = np.argmax(self.q_table[self.state])

                next_state, self.reward, done, _ = bot.step(action)
                next_state = next_state.astype(int)
                next_max = np.argmax(self.q_table[next_state])

                self.q_table[self.state, action] = self.q_table[self.state, action] + self.alpha*(self.reward +
                                         self.gamma*next_max) - self.q_table[self.state, action]

                self.state = next_state
                self.epochs += 1

            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-0.1*self.epsilon)
            if i%100 == 0:
                print("Episode: {}".format(i))

bot = MeleeBot(iso_path="/home/espen/Documents/LibMelee/Super Smash Bros. Melee (v1.02).iso")  # change to your path to melee v1.02 NTSC ISO
ql = Qlearning(0.1, 0.1)
ql.learn()
