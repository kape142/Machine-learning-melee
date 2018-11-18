import gym
import numpy as np
import random
from meleebot import MeleeBot
import time
import melee
import sys


class Qlearning:
    def __init__(self, alpha, epsilon, environment, save_qtable=True, seconds_per_episode=20):
        self.env = environment

        # Initialize the q table with shape = shape_q_table
        self.shape_q_table = (self.env.observation_space.high - self.env.observation_space.low + 1).tolist()
        self.shape_q_table.append(self.env.action_space.n)
        self.q_table1 = np.zeros(self.shape_q_table, dtype=np.float32)
        self.q_table2 = np.zeros(self.shape_q_table, dtype=np.float32)

        # Learning rate aparameters
        self.alpha = alpha                  # Synker, fra ca 0.2 og når den er 0 er læringen ferdig
        self.max_alpha = alpha
        self.min_alpha = 0
        self.decay_rate_alpha = 0.0002

        # Epsilon parameter
        self.epsilon = epsilon              # Synker, ca 1 til 0
        self.max_epsilon = epsilon
        self.min_epsilon = 0.05
        self.decay_rate = 0.0007

        # Gamma - discount rate
        self.gamma = 0.9  # hold konstant, hvor hardt du skal backtrace ting

        # Store the total reward and the cumalitive reward
        self.total_reward = [0, 0]
        self.store_cumulative_reward = [[], []]
        self.store_percentage_opponent = [[], []]
        self.animations = []
        self.episode_print = ""

        #options
        self.save_qtable = save_qtable
        self.seconds_per_episode = seconds_per_episode

    def print_data(self):
        print("Shape of the Q-table1:", self.q_table1.shape)
        print("Datatype of Q-table1:", self.q_table1.dtype)
        self.get_stored_size_q_table(self.shape_q_table)
        print()
        print("Shape of the Q-table2:", self.q_table2.shape)
        print("Datatype of Q-table2:", self.q_table2.dtype)
        self.get_stored_size_q_table(self.shape_q_table)
        print()

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
            actions["action1"] = np.ndarray.argmax(self.q_table1[state[0]])
            actions["action2"] = np.ndarray.argmax(self.q_table2[state[1]])
        return actions

    def learn(self, episode_number):
        state = self.env.reset()
        epochs = 0
        previous_percentage = [0, 0]
        current_percentage = [0, 0]
        frames = self.seconds_per_episode * 60
        done = False

        # Oppdaterer epsilon og alpha. Eksonensiell reduksjon.
        # self.epsilon = self.calculate_epsilon(episode_number)
        # self.alpha = self.calculate_alpha(episode_number)

        self.episode_print += "Epsilon: {0}\nAlpha: {1}\n\n".format(self.epsilon, self.alpha)
        actions = {"action1": 0, "action2": 0}

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

            for anim in [animations[0], animations[1]]:
                if not anim in self.animations:
                    self.animations.append(anim)

            # Want the next_state on the from [(x,y,z),(x,y,z)] with integers
            next_state[0] = tuple(next_state[0].astype(int))
            next_state[1] = tuple(next_state[1].astype(int))
            next_max1 = np.max(self.q_table1[next_state[0]])
            next_max2 = np.max(self.q_table2[next_state[1]])
            state_action1 = state[0] + (actions["action1"],)
            state_action2 = state[1] + (actions["action2"],)

            # Update Q_table for both bots
            self.q_table1[state_action1] = self.q_table1[state_action1] + np.float32(
                self.alpha * (reward[0] + self.gamma * next_max1 - self.q_table1[state_action1]))
            self.q_table2[state_action2] = self.q_table2[state_action2] + np.float32(
                self.alpha * (reward[1] + self.gamma * next_max2 - self.q_table2[state_action2]))

            # Save reward for each frame
            self.total_reward[0] += reward[0]
            self.total_reward[1] += reward[1]

            # Storing the percent of both AIs
            for idx, percent_AI in enumerate([animations[2], animations[3]]):
                current_percentage[idx] += max(percent_AI - previous_percentage[idx], 0)

            previous_percentage = [animations[2], animations[3]]
            state = next_state
            epochs += 1
            if epochs % np.floor(frames/4) == 0:
                self.episode_print += self.print_epoch_state(epochs, state)

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
            self.episode_print += "WARNING: These animations are not taken into consideration in the observation space:\n"
            for anim in unused_animations:
                self.episode_print += "%s: %0.f\n" % (melee.enums.Action(anim).name, anim)
            self.episode_print += "\n"

        # Lagrer Q_tabellen og rewards
        if self.save_qtable:
            for idx in range(2):
                self.store_cumulative_reward[idx].append(self.total_reward[idx])
                self.store_percentage_opponent[idx].append(current_percentage[idx])
            print("current_percentage_opponent: ", current_percentage)

            save_start = time.time()
            np.save('Stored_results/Q_table1_'+stored_filename+'.npy', self.q_table1)
            np.save('Stored_results/Q_table2_'+stored_filename+'.npy', self.q_table2)
            np.save('Stored_results/Rewards_'+stored_filename+'.npy', self.store_cumulative_reward)
            np.save('Stored_results/Percentage_'+stored_filename+'.npy', self.store_percentage_opponent)
            # print("Datatype of Q-table after learning:", self.q_table.dtype)
            save_time = time.time()-save_start
            self.episode_print += "Q-table and cumulative reward saved to folder 'Stored_results' with postfix '{0}.npy'"\
                             " in {1:.3f} seconds\n".format(stored_filename, save_time)
        return self.epsilon, self.alpha

    def print_epoch_state(self, epochs, state):
        string = "Epochs: {0}\n".format(epochs)
        for i in range(2):
            string += "Bot {0}'s State: {1}, Reward {2}\n".format(i,state[i],self.total_reward[i])
        return string+"\n"

    def calculate_epsilon(self, episode_number):
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate*episode_number)

    def calculate_alpha(self, episode_number):
        return self.min_alpha + (self.max_alpha - self.min_alpha) * np.exp(-self.decay_rate_alpha * episode_number)

    def reset(self):
        self.total_reward = [0, 0]
        self.episode_print = ""


def redirect_print(stored_filename):
    orig_stdout = sys.stdout
    f = open('Stored_results/console_{0}.txt'.format(stored_filename), 'a')
    sys.stdout = f
    return f, orig_stdout


def close_print(out, f):
    sys.stdout = out
    f.close()


def exit(total_episodes, start_time, out=None, f=None):
    if total_episodes > 0:
        print("Episodes per hour: {0}\n".format(total_episodes/((time.time()-start_time)/3600)))
    else:
        print("Something went wrong\n")
    print("Shutting down...\n")
    if print_to_file:
        close_print(out, f)


if __name__ == '__main__':
    bot, out, f = None, None, None
    epsilon = 1.0                   # Ratio of random actions
    alpha = 0.2                     # Learning rate
    seconds_per_episode = 20        # Seconds in game before termination
    load_old_data = True            # Load previous model and train upon this?
    save_new_data = True            # Save newly generated model for future use?
    print_to_file = True            # Print to file instead of console?
    stored_filename = 'nov18-2Qtables-Benchmark-v2'       # Postfix of the stored model after game end.
    start_time = time.time()
    episodes_to_run = 10_000
    start_episode = 0
    episode = 0
    if print_to_file:
        f, out = redirect_print(stored_filename)
    try:
        bot = MeleeBot(iso_path="melee.iso", player_control=False)
        bot.reset()
        print("Starting up...\n")
        ql = Qlearning(alpha, epsilon, bot, save_new_data, seconds_per_episode, )
        if load_old_data:
            try:
                ql.q_table1 = np.load('Stored_results/Q_table1_{0}.npy'.format(stored_filename)).astype(dtype=np.float32)
                ql.q_table2 = np.load('Stored_results/Q_table2_{0}.npy'.format(stored_filename)).astype(dtype=np.float32)
            except FileNotFoundError:
                pass
            try:
                ql.store_cumulative_reward = np.load('Stored_results/Rewards_{0}.npy'.format(stored_filename)).tolist()
                ql.store_percentage_opponent = np.load('Stored_results/Percentage_{0}.npy'.format(stored_filename)).tolist()
            except FileNotFoundError:
                pass
        start_episode = len(ql.store_cumulative_reward[0])
        if start_episode == 0:
            ql.print_data()
        if start_episode >= episodes_to_run:
            raise Exception("The bot you are starting has already finished training")
        for episode in range(start_episode, episodes_to_run):
            print_data = "============ EPISODE: {0} ============\n\n".format(episode+1)
            print_data += "Episode started {0}\n".format(str(time.strftime("%d.%b kl. %H:%M:%S ")))
            if print_to_file:
                out.write("\r{0} episodes finished".format(episode))
                out.flush()
            ql.learn(episode)
            print_data +=ql.episode_print
            ql.reset()
            print(print_data+"\n============ EPISODE END ============\n\n")
        if print_to_file:
            out.write("All {0} episodes finished".format(episodes_to_run))
            out.flush()
        exit(episodes_to_run, start_time, out, f)
        bot.dolphin.terminate()
        time.sleep(0.5)
        bot.dolphin.terminate()
    except Exception as e:
        bot.dolphin.terminate()
        time.sleep(0.5)
        bot.dolphin.terminate()
        exit(episode-start_episode, start_time, out, f)
        if not str(e) == "Dolphin is not responding":
            raise e
        else:
            print("\nDolphin is not responding, closing down...")
