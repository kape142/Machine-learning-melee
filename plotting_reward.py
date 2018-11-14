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
        newlist.append(np.mean(list[low:high]))
    return newlist


def average(list, elements_per_point=10):
    avglist = []
    for i in range(int(np.ceil(len(list)/elements_per_point))):
        avglist.append(np.mean(list[i*elements_per_point:min((i+1)*elements_per_point, len(list))]))
    return avglist


reward = np.load('Stored_results/Rewards_nov14.npy')

plt.figure(1)
noise_elements_per_point = 1
avg_elements_per_point = 1

print(len(reward[0].tolist()))
# plt.plot(reduce_noise(reward[1].tolist(), noise_elements_per_point))
for i in range(2):
    graph = average(reduce_noise(reward[i].tolist(), noise_elements_per_point), avg_elements_per_point)
    plt.plot(np.linspace(0, len(reward[i].tolist()), len(graph)), graph, label="AI {0}".format(i))
# plt.plot(average(reduce_noise(reward[1].tolist(), noise_elements_per_point), avg_elements_per_point), label="AI 2")
plt.legend()
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.show()

