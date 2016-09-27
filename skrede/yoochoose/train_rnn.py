import tensorflow as tf
import pickle
import numpy as np

n_nodes_hl1 = 2000

data = []
with open('sessions.pickle','rb') as f:
	data = pickle.load(f)

n_classes = data[0][0]
n_total_sessions = len(data)
training_data = data[:0.9*n_total_sessions]
test_data = data[0.9*n_total_sessions:]

print("training", len(training_data))
print("testing", len(testing))

batch_size = 32
total_batches = int(len(training_data)/batch_size)