import gym
import numpy as np
import random
from meleebot import MeleeBot
import time

class Qlearning:
    def __init__(self, alpha, epsilon, environment):
        self.env = environment

        # Initialize the q table with shape = shape_q_table
        shape_q_table = (self.env.observation_space.high - self.env.observation_space.low + 1).tolist()
        shape_q_table.append(self.env.action_space.n)
        self.q_table = np.zeros(shape_q_table, dtype=np.float32)

        print("Shape of the Q-table:", self.q_table.shape)
        print("Datatype of Q-table:", self.q_table.dtype)

        # Calculating the expected storage size of q_table
        stored_size = 1
        for element in shape_q_table:
            stored_size *= element
        print("Current size of Q_table: ", stored_size)
        stored_size *= 4/1e6
        print("With current size of Q-table the expected stored value is:", stored_size, "MB")

        # Learning rate aparameters
        self.alpha = alpha                  # Synker, fra ca 0.2 og når den er 0 er læringen ferdig
        self.max_alpha = 0.2
        self.min_alpha = 0.001
        self.decay_rate_alpha = 0.005

        # Epsilon parameter
        self.epsilon = epsilon              # Synker, ca 1 til 0
        self.max_epsilon = 0.99
        self.min_epsilon = 0.001
        self.decay_rate = 0.01

        # Gamma - discount rate
        self.gamma =  0.9   # hold konstant, hvor hardt du skal backtrace ting

        # Store the total reward and the cumalitive reward
        self.total_reward = [0,0]
        self.store_cumulative_reward = [[0],[0]]

    def learn(self):
        state = self.env.reset()
        epochs = 0
        done = False
        actions = {"action1":0, "action2":0}
        print("Epsilon: ", self.epsilon)
        print("Alpha: ", self.alpha)
        #print(self.q_table)

        # Want the states on the from [(x,y,z),(x,y,z)] with integeres
        for idx, states in enumerate(state):
            state[idx] = tuple(states.astype(int))

        while not done:
            if random.uniform(0,1) < self.epsilon:
                for i in range(len(actions)):
                    actions["action{0}".format(i+1)] = self.env.action_space.sample()
            else:
                for idx, states in enumerate(state):
                    actions["action{0}".format(idx+1)] = np.ndarray.argmax(self.q_table[states])
                    # if actions["action{0}".format(idx+1)] != 0:
                    #     print("Bot {0}: ".format(idx+1), "Epoch:", epochs,"Reward: ",reward, "Action: ", actions["action{0}".format(idx+1)])

            # Get the next state and reward with current aciton
            next_state, reward, done, _ = self.env.step(actions["action1"], actions["action2"])

            # Want the next_state on the from [(x,y,z),(x,y,z)] with integeres
            for idx, states in enumerate(next_state):
                next_state[idx] = tuple(states.astype(int))
                next_max = np.max(self.q_table[next_state[idx]])
                state_action = state[idx] + (actions["action{0}".format(idx+1)],)


                # Update Q_table for both bots
                self.q_table[state_action] = self.q_table[state_action] + np.float32(self.alpha * (reward[idx] + self.gamma*next_max - self.q_table[state_action]))

                # Save reward for each frame
                self.store_cumulative_reward[idx].append(self.total_reward[idx] + reward[idx])
                self.total_reward[idx] += reward[idx]



            state = next_state
            epochs += 1
            if epochs %1000 == 0:
                print("Epochs: ", epochs)
                print("Bot1's State: ", state[0], "Reward", self.total_reward[0])
                print("Bot2's State: ", state[1], "Reward", self.total_reward[1])

        done = False

        # Oppdaterer epsilon. Eksonensiell reduksjon.
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate*self.epsilon)
        self.alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * np.exp(-self.decay_rate_alpha*self.alpha)


        # Lagrer Q_tabellen
        np.save('q_table_v6_augm_stateNactions.npy', self.q_table)
        np.save('Rewards_v6.npy', self.store_cumulative_reward)
        print("Datatype of Q-table after learning:", self.q_table.dtype)
        print("Q table saved to file 'q_table_v4_augmented_states.npy'")

        return self.epsilon, self.alpha


if __name__ == '__main__':
    bot = None
    epsilon = 0.0
    alpha = 0.05
    load_old_qtable = True
    try:
        for i in range(10000):
            print("============ ITERATION: {0} ============".format(i+1))
            bot = MeleeBot(iso_path="/home/espen/Documents/TTAT3025-ML/LibMelee/Machine-learning-melee/melee.iso", player_control=False)  # change to your path to melee v1.02 NTSC ISO
            #print("Action space: ", bot.action_space.n)
            #print("Observation space: ", bot.observation_space.shape)
            #print("Epoch, reward og actions blir bare printet hvis action ut fra Q_table er noe annet enn 0! Vill skje mer flittig senere ut i treningen")
            ql = Qlearning(alpha, epsilon, bot)

            if load_old_qtable:
                ql.q_table = np.load('q_table_v6_augm_stateNactions.npy').astype(dtype=np.float32)
                #print("Type of loaded q_table: ", ql.q_table.dtype)
            bot.reset()
            while bot.CheckGameStatus == False:
                action = bot.action_space.sample()
                action2 = bot.action_space.sample()
                obv, reward, done, info = bot.step(action, action2)
            epsilon, alpha = ql.learn()

            time.sleep(1)
            bot.dolphin.terminate()
            time.sleep(0.5)
            bot.dolphin.terminate()
            time.sleep(1)
            print("\n============ ITERATION END ============\n\n")
    except Exception as e:
        print(e)
        bot.dolphin.terminate()
        time.sleep(0.5)
        bot.dolphin.terminate()
        raise e
