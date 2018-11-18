import numpy as np
import matplotlib.pyplot as plt

# def discretize_position(position):
#     sigdist = np.sign(position)
#     if abs(position) > 100:
#         discretized_position = 10*sigdist
#     else:
#         discretized_position = int(position/10)
#     return discretized_position
#
# print(discretize_position(12412))
# print(discretize_position(-23))
# print(discretize_position(123))
# print(discretize_position(-3656))
# print(discretize_position(-123))


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
    for i in range(int(np.floor(len(list)/elements_per_point))):
        avglist.append(np.mean(list[i*elements_per_point:(i+1)*elements_per_point]))
    return avglist


stored_filename = 'nov18-2Qtables-Benchmark-v2'

reward = np.load('Stored_results/Rewards_'+stored_filename+'.npy')

plt.figure(1)
noise_elements_per_point = 40
avg_elements_per_point = 1

print(len(reward[0].tolist()))
# plt.plot(reduce_noise(reward[1].tolist(), noise_elements_per_point))
for i in range(2):
    graph = average(reduce_noise(reward[i].tolist(), noise_elements_per_point), avg_elements_per_point)
    plt.plot(np.linspace(0, len(reward[i].tolist()), len(graph)), graph, label="AI {0}".format(i+1))
plt.legend()
plt.ylabel('Reward')
plt.xlabel('Episode')
#plt.show()


reward = np.load('Stored_results/Percentage_'+stored_filename+'.npy')

plt.figure(2)
noise_elements_per_point = 40
avg_elements_per_point = 1

# plt.plot(reduce_noise(reward[1].tolist(), noise_elements_per_point))
for i in range(2):
    graph = average(reduce_noise(reward[i].tolist(), noise_elements_per_point), avg_elements_per_point)
    plt.plot(np.linspace(0, len(reward[i].tolist()), len(graph)), graph, label="AI {0}".format(i+1))
plt.legend()
plt.ylabel('Percentage Opponent')
plt.xlabel('Episode')
plt.show()
