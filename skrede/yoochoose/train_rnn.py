import tensorflow as tf
import pickle
import numpy as np

n_nodes_hl1 = 2000

data = []
with open('sessions.pickle','rb') as f:
	data = pickle.load(f)

item_vec_size = -1
with open('item_vec_size.txt') as f:
    content = f.readlines()
    item_vec_size = int(content[0])
if item_vec_size == -1:
	raise Exception('Input vector size (item vector length) not set')

n_classes = data[0][0]
n_total_sessions = len(data)
slice_point = int(0.9*n_total_sessions)
train_data = data[:slice_point]
test_data = data[slice_point:]

print("training", len(train_data))
print("testing", len(test_data))

batch_size = 32
total_batches = int(len(train_data)/batch_size)
epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum':n_nodes_hl1,
				  'weight':tf.Variable(tf.random_normal([item_vec_size, n_nodes_hl1])),
				  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

output_layer = {'f_fum':None,
				'weight':tf.Variable(tf.random_normal([n_nodes_hl1, item_vec_size])),
				'bias':tf.Variable(tf.random_normal([item_vec_size]))}

def neural_network_model(inn_data):
	l1 = tf.add(tf.matmul(inn_data, hidden_1_layer['weight']), hidden_1_layer['bias'])
	l1 = tf.nn.relu(l1)
	output = tf.matmul(l1, output_layer['weight']) + output_layer['bias']
	return output

saver = tf.train.Saver()
tf_log = 'tf_log'

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

train_neural_network(x)