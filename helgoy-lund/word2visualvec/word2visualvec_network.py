import tensorflow as tf
from data_helper import generate_data
from sklearn.metrics import mean_squared_error

data_x, data_y = generate_data()

trainig_test_ratio = 0.95

training_data_x = data_x[:int(len(data_x) * trainig_test_ratio)]
training_data_y = data_y[:int(len(data_y) * trainig_test_ratio)]

test_data_x = data_x[:int(len(data_x) * 1 - trainig_test_ratio)]
test_data_y = data_y[:int(len(data_y) * 1 - trainig_test_ratio)]

# Parameters
learning_rate = 0.01
training_epochs = 5
batch_size = 500
display_step = 1

# Network parameters
n_hidden_1 = 512
n_hidden_2 = 1024
n_input = 128  # Size of Word2VisualVec vectors
n_output = 2048  # Size of image vectors

# Tensorflow graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])


def compare_vectors(v1, v2):
	return mean_squared_error(v1, v2)

# Create model
def multilayer_perceptron(x, weights, biases):
	# Hidden layer with RELU activation
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1 = tf.nn.relu(layer_1)

	# Hidden layer with RELU activation
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	layer_2 = tf.nn.relu(layer_2)

	# Output layer with linear activation
	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	return out_layer


# Store layers weight & bias
weights = {
	'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
}
biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_output]))
}

# Construct model
prediction_tensor = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.square(prediction_tensor - y))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(cost)

# Initializing all variables
init = tf.initialize_all_variables()


# Launch the graph
def get_next_training_batch(batch_size, i, training_data_x, training_data_y):
	"""
	:param batch_size: size of each batch
	:return: returns the next batch of traning data, starting where the previous ended
	"""
	x_batch = training_data_x[i * batch_size:(i + 1) * batch_size]
	y_batch = training_data_y[i * batch_size:(i + 1) * batch_size]
	return x_batch, y_batch


def train_and_test_graph():
	with tf.Session() as sess:
		sess.run(init)

		# Training cycle
		for epoch in range(training_epochs):
			avg_cost = 0.0
			total_batch = int(len(training_data_x) / batch_size)

			# Lopp over all batches
			for i in range(total_batch):
				batch_x, batch_y = get_next_training_batch(batch_size, i, training_data_x, training_data_y)
				# Run optimization op (backprop) and cost op (to get loss value)
				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

				# Compute avarage loss
				avg_cost += c / total_batch
			# Display logs per epoch step
			if epoch % display_step == 0:
				print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
		print("Optimization Finished!")

		# Test model
		# cost = tf.reduce_mean(tf.square(pred - y))

		# correct_prediction = tf.eq(tf.argmax(pred, 1), tf.argmax(y, 1))
		#
		# Calculate accuracy
		# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		# prediction_eval = accuracy.eval({x: test_data_x, y: test_data_y})
		predictions = prediction_tensor.eval({x: test_data_x, y: test_data_y})

		y_eval = y.eval({x: test_data_x, y: test_data_y})

		guessed_vectors = []
		c = 1
		len_preds = len(predictions)
		for prediction in predictions:
			print("%s/%s" % (c, len_preds))
			best_prediction = y_eval[0]
			best_prediction_mse = compare_vectors(prediction, y_eval[0])
			for y_item in y_eval:
				mse_comparison = compare_vectors(prediction, y_item)
				if mse_comparison < best_prediction_mse:
					best_prediction = y_item
					best_prediction_mse = mse_comparison
			guessed_vectors = best_prediction
			c += 1

		correct_prediction = tf.equal(guessed_vectors, y)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		# print("Accuracy:", accuracy.eval({x: test_data_x, y: test_data_y}))
		print("Accuracy:", accuracy.eval({x: test_data_x, y: test_data_y}))


def main():

	train_and_test_graph()


main()
