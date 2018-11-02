import gym
import numpy as np
import random
from meleebot import MeleeBot
import time

class Qlearning:
    def __init__(self, learning_rate, epsilon, environment):
        self.env = environment
        # Initialize the q table with shape = shape_q_table
        shape_q_table = (self.env.high - self.env.low + 1).tolist()
        shape_q_table.append(self.env.action_space.n)
        self.q_table = np.zeros(shape_q_table)
        print("Shape of the Q-table:", self.q_table.shape)

        # Parameters for the Q alogrithm
        self.alpha = learning_rate
        self.epsilon = epsilon

        # Global variable for all methods
        self.gamma =  0.9
        self.max_epsilon = 0.99
        self.min_epsilon = 0.01
        self.decay_rate = 0.01

        self.all_epochs = []



if __name__ == '__main__':
    epsilon = 1.0
    load_old_qtable = True
    bot = MeleeBot(iso_path="/home/espen/Documents/TTAT3025-ML/LibMelee/Machine-learning-melee/melee.iso", player_control=False)  # change to your path to melee v1.02 NTSC ISO
    # print("Action space: ", bot.action_space.n)
    # print("Observation space: ", bot.observation_space.shape)
    # print("Epoch, reward og actions blir bare printet hvis action ut fra Q_table er noe annet enn 0! Vill skje mer flittig senere ut i treningen")
    ql = Qlearning(0.5,epsilon, bot)
