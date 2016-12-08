import pickle
import time

# Specify stuff here
batch_size = 200
max_epochs = 10
location = '/home/ole/recSys-pro/yoochoose/'

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
    
    for i in range(1926):
        batch_cost=0
        
        _, c = sess.run([optimizer, batch_loss], feed_dict={keep_prob:0.5})
        #print(ait.shape)
        batch_cost += c

        if i%10==0:
            print("10 batches time", str(time.time()-batch_time))
            batch_time = time.time()
            print("  cost", str(batch_cost))


    print("epoch time", str(time.time()-epoch_time))
    coord.request_stop()
    coord.join(threads)
