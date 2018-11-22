import numpy as np
import matplotlib.pyplot as plt


def reduce_noise(list, elements_per_point=10):
    newlist = []
    for i in range(len(list)):
        low = max(0, i-elements_per_point)
        high = min(len(list), i+elements_per_point)
        # print(low, high, len(list[low:high]))
        newlist.append(np.mean(list[low:high+1]))
    return newlist


def average(list, elements_per_point=10):
    avglist = []
    for i in range(int(np.floor(len(list)/min(elements_per_point, len(list))))):
        avglist.append(np.mean(list[i*elements_per_point:(i+1)*elements_per_point]))
    return avglist


def show_all(stored_filename, noise_elements_per_point, avg_elements_per_point):
    add_plot('Rewards', stored_filename, noise_elements_per_point, avg_elements_per_point, 1)
    add_plot('Percentage', stored_filename, noise_elements_per_point, avg_elements_per_point, 2, True)
    add_plot('Looping_kicks', stored_filename, noise_elements_per_point, avg_elements_per_point, 3)

    plt.show()


def add_plot(type, stored_filename, noise_elements_per_point, avg_elements_per_point, index=1, reverse=False):
    try:
        reward = np.load('Stored_results/{0}_{1}.npy'.format(type, stored_filename))
    except FileNotFoundError:
        print("No data found about '{0}'".format(type))
        return

    plt.figure(index)
    loop = range(2) if not reverse else range(1, -1, -1)
    for i in loop:
        graph = average(reduce_noise(reward[i].tolist(), noise_elements_per_point), avg_elements_per_point)
        positions = np.linspace(0, len(reward[i].tolist()), len(graph))
        if len(graph) == 1:
            positions = [0, len(reward[i].tolist())]
            graph = [graph[0], graph[0]]
        ai_number = (i + 1) if not reverse else (2 - i)
        plt.plot(positions, graph, label="AI {0}".format(ai_number))
    plt.legend()
    # plt.ylim([0, 20])
    plt.ylabel(type)
    plt.xlabel('Episode')


def add_scatterplot(type, stored_filename, avg_elements_per_point, index=1, reverse=False):
    try:
        reward = np.load('Stored_results/{0}_{1}.npy'.format(type, stored_filename))
    except FileNotFoundError:
        print("No data found about '{0}'".format(type))
        return

    plt.figure(index)
    loop = range(2) if not reverse else range(1, -1, -1)
    for i in loop:
        graph = average(reward[i].tolist(),avg_elements_per_point)
        positions = np.linspace(0, len(reward[i].tolist()), len(graph))
        if len(graph) == 1:
            positions = [0, len(reward[i].tolist())]
            graph = [graph[0], graph[0]]

        plt.scatter(positions, graph, label="AI {0}".format(i + 1))
    plt.legend()
    # plt.ylim([0, 20])
    plt.ylabel(type)
    plt.xlabel('Episode')


show_all('nov2', 300, 5)