# http://www.kdnuggets.com/2016/05/intro-recurrent-networks-tensorflow.html

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import time

base_path = '/home/ole/recSys-pro/movielens-1M/'
rnn_log = base_path+'rnn.log'
training_set = base_path + '1m-training_processed.dat'

num_hidden = 100    # Number of hidden units in the hidden layer(s)
max_length = 500    # Maximum number of clicks in a session
n_classes = 3951    # Size of representation of each click
batch_size = 50
max_epochs = 5
n_training_sessions = 6037


################## READ DATA ##################################################
# create the 1-HOT representation once and then we can just use pointers to these
# time/space trade-off. Uses about 10.6 GB!
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
        line = line.split(':')
        line = map(int, line)
        # user_id and movie_id to 0-indexing
        line[0] -= 1
        line[1] -= 1
        training_data.append(line)
end = time.time()
print "   training set loaded in " + str(end-start) + "s"

# Retrieves training session i, anc converts it to the proper format for RNN input
def get_training_session(i):
    raw_session = training_data[i]
    session = []
    # add the real clicks as 1-HOT encoded vectors
    for click in raw_session:
        #c = [0.]*n_classes
        #c[click] = 1.0
        #session.append(c)
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
    #return np.array(session_x), np.array(session_y), np.array(real_session_length)
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
        for i in range(additional_from_beginnning):
            session_x, session_y, real_session_length = get_training_session(i)
            batch_x.append(session_x)
            batch_y.append(session_y)
            session_lengths.append(real_session_length)
        training_batch_current_index = additional_from_beginning
    #return np.array(batch_x), np.array(batch_y), np.array(session_lengths)
    return batch_x, batch_y, session_lengths


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

optimizer = tf.train.AdamOptimizer().minimize(cost)
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
            start = time.time()
            batch_x, batch_y, sess_len = get_next_training_batch()
            completed_training_sessions += batch_size
            _, c = sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y, session_length:sess_len})
            end = time.time()
            print " cost: "+str(c) + " | "+str(completed_training_sessions)+"/"+str(n_training_sessions)+"  | batch optimization took " + str(end-start) + "s"
            epoch_cost += c
        print "epoch "+str(epoch)+" finished."
        print "  epoch_cost = "+str(epoch_cost)
        epoch += 1
        epoch_end = time.time()
        print "   epoch took " + str(epoch_end - epoch_start) + "s to run"

