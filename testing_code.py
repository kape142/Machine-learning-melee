import numpy as np

q_table = np.zeros([5, 5, 21])
state = np.zeros(3), np.zeros(3)

print("state: ", state, "of type", type(state))

state_l = list(state)

print("state_l: ", state_l, "of type", type(state_l))

for idx, state in enumerate(state_l):
    state_l[idx] = tuple(state.astype(int))
    print("Q_table value: ",q_table[state_l[idx]])

print("state_l: ", state_l, "of type", type(state_l))
print("state_l[0]: ", state_l[0], "of type", type(state_l[0]))

for idx, state in enumerate(state_l):
    print("Q_table value: ",q_table[state_l[idx]])

new_tuple = (1,1) + (1, )
print("new tuple: ", new_tuple, type(new_tuple))
print(q_table[(1,1) + (1,)])
print(q_table[new_tuple])




#fra next_state (step) --> [array([0., 0., 0.]), array([0., 0., 0.])] <class 'list'>
# Samme som state_list her
print("-----------------------------------------------")
q_table = np.zeros([5, 5, 21])
state = np.zeros(3), np.zeros(3)

print("state: ", state, "of type", type(state))

state_l = list(state)

print("state_l: ", state_l, "of type", type(state_l))

for idx, state in enumerate(state_l):
    state_l[idx] = tuple(state.astype(int))
    print("Q_table value: ",q_table[state_l[idx]])

print("state_l: ", state_l, "of type", type(state_l))
print("state_l[0]: ", state_l[0], "of type", type(state_l[0]))



print("-----------------------------------------------")
actions = {"action1":0, "action2":0}
print(actions["action1"], type(actions["action1"]))
aa = [actions["action1"]]
print(type(aa))

val_tuple = state_l[0] + (actions["action1"], )
list = []
print(val_tuple, type(val_tuple))
for idx, state in enumerate(state_l):
    next_max = np.max(q_table[state[idx]])
    print("next_max", next_max)
    state_action = state_l[idx] + (actions["action{0}".format(idx+1)],)
    print(state_action)
    list.append(state_l[idx])
print(list)
