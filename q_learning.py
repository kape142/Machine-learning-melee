import gym
import numpy as np
import random
from meleebot import MeleeBot
import time

class Qlearning:
    def __init__(self, learning_rate, epsilon, environment):
        self.env = environment
        self.q_table = np.zeros([5, 5, 21, self.env.action_space.n])
        #print("Shape of the Q-table:", self.q_table.shape)

        # Parameters for the Q alogrithm
        self.alpha = learning_rate
        self.epsilon = epsilon

        # Global variable for all methods
        self.gamma =  0.9
        self.max_epsilon = 0.99
        self.min_epsilon = 0.01
        self.decay_rate = 0.01

        self.all_epochs = []

    def learn(self):
        state = self.env.reset()
        epochs, reward = 0, 0
        done = False
        actions = {"action1":0, "action2":0}
        print(self.q_table)

        # Want the states on the from [(x,y,z),(x,y,z)] with integeres
        state = list(state)
        for idx, states in enumerate(state):
            state[idx] = tuple(states.astype(int))

        while not done:
            if random.uniform(0,1) < self.epsilon:
                for i in range(len(actions)):
                    actions["action{0}".format(i+1)] = self.env.action_space.sample()
            else:
                for idx, states in enumerate(state):
                    actions["action{0}".format(idx+1)] = np.ndarray.argmax(self.q_table[states])
                    if actions["action{0}".format(idx+1)] != 0:
                        print("Bot {0}: ".format(idx+1), "Epoch:", epochs,"Reward: ",reward, "Action: ", actions["action{0}".format(idx+1)])
            next_state, reward, done, _ = self.env.step(actions["action1"], actions["action2"])

            # Want the next_state on the from [(x,y,z),(x,y,z)] with integeres
            for idx, states in enumerate(next_state):
                next_state[idx] = tuple(states.astype(int))
                next_max = np.max(self.q_table[next_state[idx]])
                state_action = state[idx] + (actions["action{0}".format(idx+1)],)

                # Update Q_table for both bots
                self.q_table[state_action] = self.q_table[state_action] + self.alpha * (reward[idx] + self.gamma*next_max - self.q_table[state_action])

            state = next_state
            epochs += 1

        done = False

        # ------ Inkluderer senere når jeg tar med flere episoder(flere game over) ------ #
        #self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-0.1*self.epsilon)
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate*self.epsilon)
        # Lagrer Q_tabellen
        np.save('q_table_new.npy', self.q_table)
        print("Q table saved to file 'q_table.npy'")



if __name__ == '__main__':
    bot = None
    epsilon = 0.1
    load_old_qtable = True
    try:
        for i in range(10000):
            print("Iteration: ", i+1)
            bot = MeleeBot(iso_path="/home/espen/Documents/TTAT3025-ML/LibMelee/Machine-learning-melee/melee.iso", player_control=False)  # change to your path to melee v1.02 NTSC ISO
            # print("Action space: ", bot.action_space.n)
            # print("Observation space: ", bot.observation_space.shape)
            # print("Epoch, reward og actions blir bare printet hvis action ut fra Q_table er noe annet enn 0! Vill skje mer flittig senere ut i treningen")
            ql = Qlearning(0.1,epsilon, bot)

            if load_old_qtable:
                ql.q_table = np.load('q_table.npy')

            bot.reset()
            while bot.CheckGameStatus == False:
                action = bot.action_space.sample()
                action2 = bot.action_space.sample()
                obv, reward, done, info = bot.step(action, action2)
            ql.learn()

            time.sleep(1)
            bot.dolphin.terminate()
            time.sleep(0.5)
            bot.dolphin.terminate()
            time.sleep(1)
    except Exception as e:
        print(e)
        bot.dolphin.terminate()
        time.sleep(0.5)
        bot.dolphin.terminate()
        raise e
