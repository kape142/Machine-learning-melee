import numpy as np
import matplotlib.pyplot as plt

def discretize_position(position):
    sigdist = np.sign(position)
    if abs(position) > 100:
        discretized_position = 10*sigdist
    else:
        discretized_position = int(position/10)
    return discretized_position

print(discretize_position(12412))
print(discretize_position(-23))
print(discretize_position(123))
print(discretize_position(-3656))
print(discretize_position(-123))




reward = np.load('Rewards_model-v1.npy')
#print(reward[0].tolist())



plt.plot(reward[0].tolist())
plt.ylabel('some numbers')
plt.show()
