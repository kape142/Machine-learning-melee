import gym
import numpy as np
import random
from meleebot import MeleeBot
import time
import melee
import sys


class Qlearning:
    def __init__(self, alpha, epsilon, environment, save_qtable=True):
        self.env = environment

        # Initialize the q table with shape = shape_q_table
        shape_q_table = (self.env.observation_space.high - self.env.observation_space.low + 1).tolist()
        shape_q_table.append(self.env.action_space.n)
        self.q_table = np.zeros(shape_q_table, dtype=np.float32)

        # Print information about th Q table
        print("Shape of the Q-table:", self.q_table.shape)
        print("Datatype of Q-table:", self.q_table.dtype)
        self.get_stored_size_q_table(shape_q_table)
        print()

        # Learning rate aparameters
        self.alpha = alpha                  # Synker, fra ca 0.2 og når den er 0 er læringen ferdig
        self.max_alpha = alpha
        self.min_alpha = 0.0003
        self.decay_rate_alpha = 0.0003      # Denne burde vel være veldig lav for at den skal kunne lære lenge?

        # Epsilon parameter
        self.epsilon = epsilon              # Synker, ca 1 til 0
        self.max_epsilon = epsilon
        self.min_epsilon = 0.001
        self.decay_rate = 0.001

        # Gamma - discount rate
        self.gamma = 0.9  # hold konstant, hvor hardt du skal backtrace ting

        # Store the total reward and the cumalitive reward
        self.total_reward = [0, 0]
        self.store_cumulative_reward = [[0],[0]]
        self.animations = []

        #options
        self.save_qtable = save_qtable

    def get_stored_size_q_table(self, shape_q_table):
        # Calculating the expected storage size of q_table
        stored_size = 1
        for element in shape_q_table:
            stored_size *= element
        print("Current size of Q_table: ", stored_size)
        stored_size *= 4 / 1e6
        print("With current size of Q-table the expected stored value is:", stored_size, "MB")

    def get_action(self, actions, state):
        if random.uniform(0, 1) < self.epsilon:
            for i in range(len(actions)):
                actions["action{0}".format(i + 1)] = self.env.action_space.sample()
        else:
            for idx, states in enumerate(state):
                actions["action{0}".format(idx + 1)] = np.ndarray.argmax(self.q_table[states])
                # if actions["action{0}".format(idx+1)] != 0:
                #       print("Bot {0}: ".format(idx+1), "Epoch:", epochs,"Reward: ",reward,
                #             "Action: ", actions["action{0}".format(idx+1)])
        return actions

    def learn(self, seconds):
        state = self.env.reset()
        epochs = 0
        frames = seconds * 60
        done = False

        actions = {"action1": 0, "action2": 0}
        print("Epsilon: ", self.epsilon)
        print("Alpha: ", self.alpha)
        print()
        # print(self.q_table)

        # Want the states on the from [(x,y,z),(x,y,z)] with integeres
        for idx, states in enumerate(state):
            state[idx] = tuple(states.astype(int))

        # Wait until game have started before updating the Q_table
        while not self.env.in_game:
            self.env.step(actions["action1"], actions["action2"])

        animations = (323, 323)
        while animations[0] in [322, 323, 324] or animations[1] in [322, 323, 324]:
            _, _, _, animations = self.env.step(actions["action1"], actions["action2"])

        # while not done:   # Game continues until game over (include done false before and after while loop)
        for epoch in range(1, frames+1):
            # Get random action or action from Q table.
            actions = self.get_action(actions, state)

            # Get the next state and reward with current aciton
            next_state, reward, done, animations = self.env.step(actions["action1"], actions["action2"])

            for anim in animations:
                if not anim in self.animations:
                    self.animations.append(anim)
            # Want the next_state on the from [(x,y,z),(x,y,z)] with integers
            for idx, states in enumerate(next_state):
                next_state[idx] = tuple(states.astype(int))
                next_max = np.max(self.q_table[next_state[idx]])
                state_action = state[idx] + (actions["action{0}".format(idx + 1)],)

                # Update Q_table for both bots
                self.q_table[state_action] = self.q_table[state_action] + np.float32(
                    self.alpha * (reward[idx] + self.gamma * next_max - self.q_table[state_action]))

                # Save reward for each frame
                self.total_reward[idx] += reward[idx]

            state = next_state
            epochs += 1
            if epochs % np.floor(frames/4) == 0:
                self.print_epoch_state(epochs, state)


        self.env.done = True
        while not done:
            _, _, done, _ = self.env.step(0, 0)
        # done = False
        self.animations.sort()
        unused_animations = []
        for anim in self.animations:
            if bot.action_to_number(anim) < 0:
                unused_animations.append(anim)
        if len(unused_animations) > 0:
            print("WARNING: These animations are not taken into consideration in the observation space:")
            for anim in unused_animations:
                print("%s: %0.f" % (melee.enums.Action(anim).name, anim))
            print()

        # Oppdaterer epsilon og alpha. Eksonensiell reduksjon.
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate*(episode))
        self.alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * np.exp(-self.decay_rate_alpha*(episode))

        # Lagrer Q_tabellen og rewards
        if self.save_qtable:
            self.store_cumulative_reward[0].append(self.total_reward[0])
            self.store_cumulative_reward[1].append(self.total_reward[1])

            np.save('Stored_results/Q_table_'+stored_filename+'.npy', self.q_table)
            np.save('Stored_results/Rewards_'+stored_filename+'.npy', self.store_cumulative_reward)
            # print("Datatype of Q-table after learning:", self.q_table.dtype)
            print("Q-table and cumulative reward saved to folder 'Stored_results' with postfix '{0}.npy'\n"
                  .format(stored_filename))

        return self.epsilon, self.alpha

    def print_epoch_state(self, epochs, state):
        print("Epochs: ", epochs)
        print("Bot1's State: ", state[0], "Reward", self.total_reward[0])
        print("Bot2's State: ", state[1], "Reward", self.total_reward[1])
        print()

    def reset(self):
        self.total_reward = [0, 0]


def redirect_print(stored_filename):
    orig_stdout = sys.stdout
    f = open('Stored_results/console_{0}.txt'.format(stored_filename), 'w')
    sys.stdout = f
    return f, orig_stdout


def close_print(out, f):
    sys.stdout = out
    f.close()


def exit(total_episodes, start_time, out=None, f=None):
    print("Episodes per hour: {0}\n".format(total_episodes/((time.time()-start_time)/3600)))
    if print_to_file:
        close_print(out, f)


if __name__ == '__main__':
    bot, out, f = None, None, None
    epsilon = 0.99                  # Ratio of random actions
    alpha = 0.2                     # Learning rate
    seconds = 20                    # Seconds in game before termination
    load_old_qtable = True         # Load previous model and train upon this?
    save_new_qtable = True          # Save newly generated model for future use?
    print_to_file = True           # Print to file instead of console?
    stored_filename = 'nov13'       # Postfix of the stored model after game end.
    start_time = time.time()
    episode = 0
    if print_to_file:
        f, out = redirect_print(stored_filename)
    try:
        bot = MeleeBot(iso_path="melee.iso", player_control=False)
        bot.reset()

        ql = Qlearning(alpha, epsilon, bot, save_new_qtable)

        if load_old_qtable:
            ql.q_table = np.load('Stored_results/Q_table_' + stored_filename + '.npy').astype(dtype=np.float32)
            # print("Type of loaded q_table: ", ql.q_table.dtype)

        for episode in range(1, 1001):
            print("============ EPISODE: {0} ============\n".format(episode))
            print("Episoden startet {0}\n".format(str(time.strftime("%d. %B kl. %H:%M:%S "))))
            if print_to_file:
                out.write("\r{0} episodes finished".format(episode-1))
                out.flush()
            epsilon, alpha = ql.learn(seconds)
            ql.reset()
            print("============ EPISODE END ============\n\n")
        exit(1000, start_time, out, f)
        bot.dolphin.terminate()
        time.sleep(0.5)
        bot.dolphin.terminate()
    except Exception as e:
        bot.dolphin.terminate()
        time.sleep(0.5)
        bot.dolphin.terminate()
        exit(episode-1, start_time, out, f)
        if not str(e) == "Dolphin is not responding":
            raise e
        else:
            print("\nDolphin is not responding, closing down...")

