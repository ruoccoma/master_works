from __future__ import division
from evaluation import *
from baselines import *
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import time

base_path = '/home/ole/recSys-pro/movielens-1M/'
rnn_log = base_path+'rnn.log'
training_set = base_path + '1M-train.dat'
test_set = base_path + '1M-test.dat'

num_hidden = 100    # Number of hidden units in the hidden layer(s)
max_length = 20    # Maximum number of clicks in a session
n_classes = 3952    # Size of representation of each click
batch_size = 50
max_epochs = 20

# Just decleration, updated when we read the datasets
n_training_sessions = None
n_test_sessions = None


################## READ DATA ##################################################
# create the 1-HOT representation once and then we can just use pointers to these
# time/space trade-off.
print " creating library of 1-HOT vectors..."
start = time.time()
one_hot_representations = []
for i in range(n_classes):
    tmp = [0.]*n_classes
    tmp[i] = 1.0
    one_hot_representations.append(np.array(tmp))
dummy_vector = [0.]*n_classes
end = time.time()
print "   1-HOT vectors created in " + str(end-start) + "s"

print " loading training set..."
start = time.time()
training_data = []
with open(training_set, 'r') as f:
    for line in f:
        line = line.split(' ')
        line = map(int, line)
        training_data.append(line)
end = time.time()
n_training_sessions = len(training_data)
print "   training set loaded in " + str(end-start) + "s"

print " loading test set..."
start = time.time()
test_data = []
with open(test_set, 'r') as f:
    for line in f:
        line = line.split(' ')
        line = map(int, line)
        test_data.append(line)
end = time.time()
n_test_sessions = len(test_data)
print "   test set loaded in " + str(end-start) + "s"

# Retrieves training session i, anc converts it to the proper format for RNN input
def get_training_session(i):
    raw_session = training_data[i]
    session = []
    # add the real clicks as 1-HOT encoded vectors
    for click in raw_session:
        session.append(one_hot_representations[click])
    # Might need to pad the session to reach max_length
    padding_length = max_length - len(session)
    real_session_length = len(session) - 1  # last one is only target
    for i in range(padding_length):
        session.append(dummy_vector)
    
    # Split session into input and target clicks
    session_x = []
    session_y = []
    for i in range(len(session)-1):
        session_x.append(session[i])
        session_y.append(session[i+1])
    return session_x, session_y, real_session_length

# Method to easily get next training batch
training_batch_current_index = 0
def get_next_training_batch():
    global training_batch_current_index
    batch_x = []
    batch_y = []
    session_lengths = []
    if training_batch_current_index + batch_size < len(training_data):
        # Either we can get the batch_size next batches
        for i in range(training_batch_current_index, training_batch_current_index+batch_size):
            session_x, session_y, real_session_length = get_training_session(i)
            batch_x.append(session_x)
            batch_y.append(session_y)
            session_lengths.append(real_session_length)
        training_batch_current_index += batch_size
    else:
        # or we will reach the end of the set, and need to start from the beginning
        additional_from_beginning = batch_size - (len(training_data)-training_batch_current_index)
        for i in range(training_batch_current_index, len(training_data)):
            session_x, session_y, real_session_length = get_training_session(i)
            batch_x.append(session_x)
            batch_y.append(session_y)
            session_lengths.append(real_session_length)
        for i in range(additional_from_beginning):
            session_x, session_y, real_session_length = get_training_session(i)
            batch_x.append(session_x)
            batch_y.append(session_y)
            session_lengths.append(real_session_length)
        training_batch_current_index = additional_from_beginning
    return batch_x, batch_y, session_lengths

test_sample_index = 0
def get_next_test_batch():
    global test_sample_index
    test_batch_x = []
    test_batch_y = []
    test_batch_lengths = []
    batch_index_goal = test_sample_index + batch_size
    
    while test_sample_index < n_test_sessions and test_sample_index < batch_index_goal:
        tmp_x, tmp_y, tmp_s = get_test_sample(test_sample_index)
        test_batch_x.append(tmp_x)
        test_batch_y.append(tmp_y)
        test_batch_lengths.append(tmp_s)
        test_sample_index += 1
    
    test_set_completed = False
    if n_test_sessions <= test_sample_index:
        test_set_completed = True
        test_sample_index = 0

    return test_batch_x, test_batch_y, test_batch_lengths, test_set_completed

def get_test_sample(i):
    sample_x = []
    sample_y = []
    session_lengths = []
    raw_sample = test_data[i]
    sample = []
    for click in raw_sample:
        sample.append(one_hot_representations[click])
    padding_length = max_length - len(sample)
    real_sample_length = len(sample) - 1 # last one is only target
    for i in range(padding_length):
        sample.append(dummy_vector)

    for i in range(len(sample)-1):
        sample_x.append(sample[i])
    for i in range(1, len(raw_sample)):
        sample_y.append(raw_sample[i])

    return sample_x, sample_y, real_sample_length

################## BASELINE COMPARISON ########################################

def print_baseline_random():
    accuracy = 20/n_classes
    print "| accuracy of random guessing should be about: "+str(accuracy)

def print_baseline_top_k():
    # the top-k movies are found by preprocessing.py, and can be found in the preprocess.log
    top_k_movies = [1264, 1616, 526, 1196, 2395, 109, 2761, 607, 1197, 1579, 592, 1269, 2570, 588, 2027, 479, 1209, 1195, 259, 2857]

    correct_pred = 0
    count = 0
    for session in test_data:
        for movie in session[1:]:
            if movie in top_k_movies:
                correct_pred += 1
            count += 1
    
    accuracy = correct_pred/count
    print "| accuracy of top-k baseline: " + str(accuracy)

print "----- Baselines ---------"
print_baseline_random()
print_baseline_top_k()
# Just print the result from last run for this, it does not change
#print_baseline_knn(20, training_data, test_data)
print "| accuracy of k-nn baseline: 0.200278134474 | k=20"
print "-------------------------"

################# MODEL SETUP #################################################
print " creating model..."
start = time.time()
# [Batch size, #timesteps (minus 1 since each click has the next one as target), click representation size]
x = tf.placeholder(tf.float32, [None, max_length-1, n_classes], name="input_x")
y = tf.placeholder(tf.float32, [None, max_length-1, n_classes], name="target_y")
# Vector with the session lengths for each session in a batch.
session_length = tf.placeholder(tf.int32, [None], name="seq_len_of_input")
output, state = rnn.dynamic_rnn(
        rnn_cell.GRUCell(num_hidden),
        x,
        dtype=tf.float32,
        sequence_length=session_length
        )

layer = {'weights':tf.Variable(tf.random_normal([num_hidden, n_classes])),
        'biases':tf.Variable(tf.random_normal([n_classes]))}

# Flatten to apply same weights to all time steps.
output = tf.reshape(output, [-1, num_hidden])
prediction = tf.matmul(output, layer['weights'])# + layer['biases'] TODO

# Unflatten (?) back to original shape
# skip this, since it they cancel each other out
#prediction = tf.reshape(prediction, [-1, max_length, n_classes])
#prediction = tf.reshape(prediction, [-1, n_classes])

y_flat = tf.reshape(y, [-1, n_classes])
# Reduce sum, since average divides by max_length, which is often wrong
cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(prediction,y_flat))

# The training 'function'
optimizer = tf.train.AdamOptimizer().minimize(cost)

# The testing part
top_pred_vals, top_pred_indices = tf.nn.top_k(prediction, k=20, sorted=True, name="top_predoctions")


end = time.time()
print "   model created in " + str(end-start) + " s"


################## RUN TRAINING ###############################################
init = tf.initialize_all_variables()
with tf.Session() as sess:
    print " initializing all variables..."
    start = time.time()
    sess.run(init)
    end = time.time()
    print "   variables initialized in " + str(end-start) + " s"
    
    epoch = 1
    while epoch <= max_epochs:
        epoch_start = time.time()
        completed_training_sessions = 0
        epoch_cost = 0
        while completed_training_sessions < n_training_sessions:
            batch_x, batch_y, sess_len = get_next_training_batch()
            completed_training_sessions += batch_size
            _, c = sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y, session_length:sess_len})
            epoch_cost += c
        epoch_time = time.time() - epoch_start
        print "epoch "+str(epoch)+" | cost: "+str(epoch_cost)+" | time: " + str(epoch_time) + "s"
        
        # Test the accuracy
        if epoch%1 == 0:
            test_start = time.time()
            correct_preds = 0
            count = 0
            ranks = []
            test_set_completed = False
            while not test_set_completed:
                test_x, test_y, sess_len, test_set_completed = get_next_test_batch()
                preds = sess.run([top_pred_indices], feed_dict={x:test_x, session_length:sess_len})
                for batch_i in range(len(preds)):
                    batch_preds = preds[batch_i]
                    batch_y = test_y[batch_i]
                    for i in range(len(batch_y)):
                        correct_click = batch_y[i]
                        if correct_click in batch_preds[i]:
                            correct_preds += 1
                        count += 1
                        ranks.append(get_reciprocal_rank(correct_click, batch_preds[i]))
            accuracy = correct_preds / count
            mrr = sum(ranks)/len(ranks)
            test_time = time.time() - test_start
            print "testing accuracy: "+str(accuracy)+" | MRR@20: "+str(mrr)+" | time: "+str(test_time)+"s"

        epoch += 1


