import gym
import numpy as np
import random
from meleebot import MeleeBot
import time
import melee
import sys


class QTableBenchmark:
    def __init__(self, environment, seconds_per_episode=20):
        self.env = environment

        # Initialize the q table with shape = shape_q_table
        self.shape_q_table = (self.env.observation_space.high - self.env.observation_space.low + 1).tolist()
        self.shape_q_table.append(self.env.action_space.n)
        self.q_table = np.zeros(self.shape_q_table, dtype=np.float32)

        # Store the total reward and the cumulative reward
        self.total_reward = [0, 0]
        self.store_cumulative_reward = [[], []]
        self.store_percentage_opponent = [[], []]
        self.store_looping_kick = [[], []]
        self.animations = []
        self.episode_print = ""

        #options
        self.seconds_per_episode = seconds_per_episode

    def print_data(self):
        print("Shape of the Q-table:", self.q_table.shape)
        print("Datatype of Q-table:", self.q_table.dtype)
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
        actions["action2"] = self.env.action_space.sample()
        actions["action1"] = np.ndarray.argmax(self.q_table[state])
        return actions

    def step(self):
        epochs=0
        state = self.env.reset()
        previous_percentage = [0, 0]
        current_percentage = [0, 0]
        current_looping_kicks = [0, 0]
        is_kicking = [False, False]
        kick_pre_damage = [0,0]
        frames = self.seconds_per_episode * 60
        done = False

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
            actions = self.get_action(actions, state[0])

            # Get the next state and reward with current aciton
            next_state, reward, done, animations = self.env.step(actions["action1"], actions["action2"])

            for idx, anim in enumerate([animations[0], animations[1]]):
                if self.env.action_to_number(anim) == 8:
                    if not is_kicking[idx]:
                        kick_pre_damage[idx] = current_percentage[1-idx]
                        is_kicking[idx] = True
                        current_looping_kicks[idx] += 1
                else:
                    if is_kicking[idx]:
                        print("This kick by AI {0} did {1} damage".format(idx, current_percentage[1-idx]-kick_pre_damage[idx]))
                        is_kicking[idx] = False

                if anim not in self.animations:
                    self.animations.append(anim)
            # Want the next_state on the from [(x,y,z),(x,y,z)] with integers
            for idx, states in enumerate(next_state):
                next_state[idx] = tuple(states.astype(int))
                next_max = np.max(self.q_table[next_state[idx]])
                state_action = state[idx] + (actions["action{0}".format(idx + 1)],)
                # Save reward for each frame
                self.total_reward[idx] += reward[idx]

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

        # Lagrer Q_tabellen og rewards
        for idx in range(2):
            self.store_cumulative_reward[idx].append(self.total_reward[idx])
            self.store_percentage_opponent[idx].append(current_percentage[idx])
            self.store_looping_kick[idx].append(current_looping_kicks[idx])

        save_start = time.time()
        # np.save('Stored_results/Q_table_'+stored_filename+'.npy', self.q_table)
        np.save('Stored_results/Rewards_'+stored_filename+'.npy', self.store_cumulative_reward)
        np.save('Stored_results/Percentage_'+stored_filename+'.npy', self.store_percentage_opponent)
        np.save('Stored_results/Looping_kicks_'+stored_filename+'.npy', self.store_looping_kick)
        # print("Datatype of Q-table after learning:", self.q_table.dtype)
        save_time = time.time()-save_start
        self.episode_print += "Data saved to folder 'Stored_results' with postfix '{0}.npy'"\
                         " in {1:.3f} seconds\n".format(stored_filename, save_time)

    def print_epoch_state(self, epochs, state):
        string = "Epochs: {0}\n".format(epochs)
        for i in range(2):
            string += "Bot {0}'s State: {1}, Reward {2}\n".format(i,state[i],self.total_reward[i])
        return string+"\n"

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
    epsilon = 0                      # Ratio of random actions
    seconds_per_episode = 20         # Seconds in game before termination
    print_to_file = False            # Print to file instead of console?
    stored_filename = '18k_benchmark'        # Postfix of the stored model after game end.
    start_time = time.time()
    episodes_to_run = 20_000
    start_episode = 0
    episode = 0
    if print_to_file:
        f, out = redirect_print(stored_filename)
    try:
        bot = MeleeBot(iso_path="melee.iso", player_control=False)
        bot.reset()
        print("Starting up...\n")
        ql = QTableBenchmark(bot, seconds_per_episode, )
        try:
            ql.q_table = np.load('Stored_results/Q_table_{0}.npy'.format(stored_filename)).astype(dtype=np.float32)
        except FileNotFoundError:
            pass
        try:
            ql.store_percentage_opponent = np.load('Stored_results/Percentage_{0}.npy'.format(stored_filename)).tolist()
        except FileNotFoundError:
            pass
        try:
            ql.store_cumulative_reward = np.load('Stored_results/Rewards_{0}.npy'.format(stored_filename)).tolist()
        except FileNotFoundError:
            pass
        try:
            ql.store_looping_kick = np.load('Stored_results/Looping_kicks_{0}.npy'.format(stored_filename)).tolist()
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
            ql.step()
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
