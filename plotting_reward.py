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
    for i in range(len(list)):
        list[i] = np.mean(list[max(0, i-elements_per_point):min(len(list), i+elements_per_point)])
    return list


def average(list, elements_per_point=10):
    avglist = []
    for i in range(int(np.ceil(len(list)/elements_per_point))):
        avglist.append(np.mean(list[i*elements_per_point:min((i+1)*elements_per_point, len(list))]))
    return avglist


reward = np.load('Stored_results/Rewards_nov13.npy')

plt.figure(1)
plt.plot(average(reduce_noise(reward[0].tolist(), 100)), label="AI 1")
plt.plot(average(reduce_noise(reward[1].tolist(), 100)), label="AI 2")
plt.legend()
plt.ylabel('Reward')
plt.xlabel('Epoch')
plt.show()


