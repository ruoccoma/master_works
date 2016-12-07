from __future__ import division
from evaluation import *
from baselines import *
from batch_operations import *
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import time

base_path = '/home/ole/recSys-pro/yoochoose/'
training_data_file = base_path + 'rsc15_train_full_ole.txt'
test_data_file = base_path + 'rsc15_test_ole.txt'
#knn_data_file = base_path + '1M-train-knn.dat'
preprocess_log = base_path + 'preprocess.log'

num_hidden = 100    # number of hidden units in the RNN layer
batch_size = 200
max_epochs = 10
# number of epochs between each test run
test_frequency = 1
use_ff = True

# Load some values about the dataset
max_length, n_classes, top_k = load_meta_values(preprocess_log)
print("max_length = "+str(max_length)+" | n_classes = " + str(n_classes))

dataset_manager = DatasetManager(training_data_file, test_data_file, n_classes, max_length, batch_size)

# Prints precomputed baslines. Use print_all_baselines() to recalculate
#knn_training_set = load_knn_training_set(knn_data_file)
#print_all_baselines(max_length, n_classes, dataset_manager.get_test_set(), top_k, knn_training_set)
print_all_baslines_precomputed()

# [Batch size, #timesteps (minus 1 since each click has the next one as target), click representation size]
x = tf.placeholder(tf.float32, [None, max_length-1, n_classes], name="InputX")
y = tf.placeholder(tf.float32, [None, max_length-1, n_classes], name="TargetY")
# Vector with the session lengths for each session in a batch.
session_length = tf.placeholder(tf.int32, [None], name="SeqLenOfInput")
if use_ff:
    output, state = rnn.dynamic_rnn(
        rnn_cell.GRUCell(num_hidden),
        x,
        dtype=tf.float32,
        sequence_length=session_length
        )
else:
    output, state = rnn.dynamic_rnn(
        rnn_cell.GRUCell(n_classes),
        x,
        dtype=tf.float32,
        sequence_length=session_length
        )

layer = {'weights':tf.Variable(tf.random_normal([num_hidden, n_classes])),
        'biases':tf.Variable(tf.random_normal([n_classes]))}

keep_prob = tf.placeholder(tf.float32)
# Flatten to apply same weights to all time steps.
if use_ff:
    # add dropout [no significant difference]
    #output = tf.nn.dropout(output, keep_prob)

    output = tf.reshape(output, [-1, num_hidden])
    prediction = tf.matmul(output, layer['weights'])# + layer['biases'] TODO
else:
    prediction = tf.reshape(output, [-1, n_classes])

y_flat = tf.reshape(y, [-1, n_classes])

# Reduce sum, since average divides by max_length, which is often wrong
final_output = tf.nn.softmax_cross_entropy_with_logits(prediction,y_flat)
cost = tf.reduce_sum(final_output)

# The training 'function'
optimizer = tf.train.AdamOptimizer().minimize(cost)

# The testing part
top_pred_vals, top_pred_indices = tf.nn.top_k(prediction, k=20, sorted=True, name="TopPredictions")


# Run training and testing
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    # Go through specified number of epochs
    for epoch in range(max_epochs):

        # Record runtime for each epoch and cost
        epoch_start = time.time()
        epoch_cost = 0

        # Split the epoch in batches
        num_training_batches = dataset_manager.get_number_of_training_batches()
        for training_batch_number in range(num_training_batches):
            batch_time = time.time()
            print(str(training_batch_number)+"/"+str(num_training_batches))
            batch_data_start = time.time()
            batch_x, batch_y, sess_len = dataset_manager.get_next_training_batch()
            print("  batch data retrieved in "+str(time.time()-batch_data_start)+" s")

            # Calculate loss and optimize the weights
            o, c = sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y, session_length:sess_len, keep_prob:0.5})
            epoch_cost += c
            print("batch_time: "+str(time.time()-batch_time)+" s")

        # Print statistics for training in this epoch
        epoch_time = time.time() - epoch_start
        print("epoch "+str(epoch)+" | cost: "+str(epoch_cost)+" | time: " + str(epoch_time) + "s")
        
        # Test the accuracy
        if epoch%test_frequency == 0:

            # Record runtime of testing, pluss statistics for test accuracy++
            test_start = time.time()
            correct_preds = 0
            total_num_items_count = 0
            ranks = []

            # Split test in batches
            num_test_batches = dataset_manager.get_number_of_test_batches()
            for test_batch_number in range(num_test_batches):
                test_x, test_y, sess_len = dataset_manager.get_next_test_batch()

                # Get top (20) predictions for each session
                preds = sess.run([top_pred_indices], feed_dict={x:test_x, session_length:sess_len, keep_prob:1.0})

                # For each test batch
                for batch_i in range(len(preds)):
                    # Get predictions and targets for a session in the batch
                    batch_preds = preds[batch_i]
                    batch_y = test_y[batch_i]

                    # For items in session
                    for i in range(len(batch_y)):
                        # Get the correct click aka target value
                        correct_click = batch_y[i]

                        # Check if the predictions contain the target value
                        if correct_click in batch_preds[i]:
                            correct_preds += 1

                        total_num_items_count += 1

                        # Calculate reciprocal ranks for the MeanReciprocalRank
                        reciprocal_rank = get_reciprocal_rank(correct_click, batch_preds[i])
                        ranks.append(reciprocal_rank)

            # Calculate accuracy and MRR
            accuracy = correct_preds / total_num_items_count
            mrr = sum(ranks)/len(ranks)
            
            test_time = time.time() - test_start
            print(" Recall@20: "+str(accuracy)+" | MRR@20: "+str(mrr)+" | time: "+str(test_time)+"s")
