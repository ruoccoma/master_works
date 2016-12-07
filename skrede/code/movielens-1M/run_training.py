import pickle
import time

# Specify stuff here
batch_size = 200
max_epochs = 10
location = '/home/ole/recSys-pro/movielens-1M/'

# Load and store some parameters
meta_pickle = location + 'meta.pickle'
meta_data   = pickle.load( open(meta_pickle, 'rb'))
meta_data['batch_size'] = batch_size
pickle.dump(meta_data, open(meta_pickle, 'wb'))

# Create the model
from rnn_model import *

# Run training
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    epoch_time = time.time()
    batch_time = time.time()
    epoch_cost = 0
    for i in range(1926):
        _ = sess.run([optimizer], feed_dict={keep_prob:0.5})
        epoch_cost += 0
        if i%30==0:
            print("30 batches time", str(time.time()-batch_time))
            print("  cost", str(epoch_cost))
            epoch_cost=0
            batch_time = time.time()
    print("epoch time", str(time.time()-epoch_time))
    coord.request_stop()
    coord.join(threads)
