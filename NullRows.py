import numpy as np

def get_stored_size_q_table(shape_q_table):
    # Calculating the expected storage size of q_table
    stored_size = 1
    for element in shape_q_table:
        stored_size *= element
    return stored_size

q_table1 = np.load('Stored_results/Q_table1_nov2.npy')
q_table2 = np.load('Stored_results/Q_table_18k.npy')

Q_table1_size = get_stored_size_q_table(q_table1.shape)
Q_table2_size = get_stored_size_q_table(q_table2.shape)


print("Non-zero elements in Q-table1: ", np.count_nonzero(q_table1), "/", Q_table1_size)
print("Non-zero elements in Q-table2: ", np.count_nonzero(q_table2),  "/", Q_table2_size)