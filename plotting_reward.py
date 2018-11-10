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





reward = np.load('Stored_results/Rewards_model-v1.npy')

plt.figure(1)
plt.plot(reward[0].tolist(), label="AI 1")
plt.plot(reward[1].tolist(), label="AI 2")
plt.legend()
plt.ylabel('Reward')
plt.xlabel('Epoch')
plt.show()
