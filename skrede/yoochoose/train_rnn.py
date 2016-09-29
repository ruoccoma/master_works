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
total_batches = 167
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
		try:
			epoch = int(open(tf_log,'r').read().split('\n')[-2])+1
			print('STARTING:',epoch)
		except:
			epoch = 1

		while epoch <= epochs:
			epoch_loss = 1
			batch_x = []
			batch_y = []
			batches_run = 0
			for session in train_data:
				for i in range(len(session)-1):
					item_x = session[i]
					item_y = session[i+1]
					batch_x.append(item_x)
					batch_y.append(item_y)

					if len(batch_x) >= batch_size:
						_, c = sess.run([optimizer, cost], feed_dict={x: np.array(batch_x),
																	  y: np.array(batch_y)})
						epoch_loss += c
						batch_x = []
						batch_y = []
						batches_run += 1
						print('Batch run:', batches_run, '/', total_batches, '| Epoch:', epoch, '| Batch loss:', c)
			saver.save(sess, "model.ckpt")
			print('Epoch', epoch, 'completed out of',epochs,'loss:',epoch_loss)
			with open(tf_log,'w') as f:
				f.write(str(epoch)+'\n')
			epoch += 1

#train_neural_network(x)

def test_neural_network():
	prediction = neural_network_model(x)
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		try:
			saver.restore(sess, "model.ckpt")
		except Exception as e:
			print(str(e))

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		test_x = []
		test_y = []
		counter = 0
		for session in test_data:
			for i in range(len(session)-1):
				test_x.append(session[i])
				test_y.append(session[i+1])
				counter += 1
		print('Tested', counter, 'samples')
		test_x = np.array(test_x)
		test_y = np.array(test_y)
		print('Accuracy:', accuracy.eval({x:test_x, y:test_y}))


test_neural_network()