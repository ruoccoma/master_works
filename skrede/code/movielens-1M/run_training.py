import pickle
import time
import numpy as np
from data_handler import *

# Specify stuff here
batch_size = 200
max_epochs = 50
location = '/home/ole/recSys-pro/movielens-1M/'
test_file = location + '1M-test.csv'

# Load and store some parameters
meta_pickle = location + 'meta.pickle'
meta_data   = pickle.load( open(meta_pickle, 'rb'))
meta_data['batch_size'] = batch_size
meta_data['max_epochs'] = max_epochs

n_classes           = meta_data['n_classes']
n_training_examples = meta_data['n_training_examples']
k                   = meta_data['k']
pickle.dump(meta_data, open(meta_pickle, 'wb'))

test_data = load_data_set(test_file)

# Create the model
from rnn_model import *

print('')
print("model created, starting training with the following parameters")
print("  max_epochs:", max_epochs, "| batch_size:", batch_size, "| data files from", location)
print('')

# Run training
init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for e in range(max_epochs):
        epoch_time = time.time()
        epoch_loss = 0
    
        for b in range(int(n_training_examples/batch_size)):
            _, l = sess.run([optimizer, batch_loss], feed_dict={keep_prob:0.5})
            epoch_loss += l

        print("epoch #"+str(e),"finished in", str(time.time()-epoch_time), "s")
        print("  epoch loss", epoch_loss)

        print("testing...")
        bt = time.time()
        test_batch = test_data[:200]
        sl = [e[0] for e in test_batch]
        xu = [e[1] for e in test_batch]
        yu = [e[2] for e in test_batch]
        m = [e[3] for e in test_batch]
        print("  test_batch created in", str(time.time()-bt))

        preds = sess.run(top_k_preds, feed_dict={keep_prob:1.0, session_length:sl, x_unparsed:xu, mask:m})

        # Calculate accuracy
        recall, mrr = 0.0, 0.0
        evaluation_count = 0
        for batch_index in range(preds.shape[0]):
            pred_sequence = preds[batch_index]
            target_sequence = yu[batch_index]

            for i in range(sl[batch_index]):
                target_item = target_sequence[i]
                k_predictions = pred_sequence[i]

                if target_item in k_predictions:
                    recall += 1
                    rank = np.nonzero(k_predictions == target_item)[0][0]+1
                    mrr += 1.0/rank
                
                evaluation_count += 1
        recall = recall/evaluation_count
        mrr = mrr/evaluation_count

        print("  Recall@"+str(k)+":", recall)
        print("  MRR@"+str(k)+":", mrr)

    coord.request_stop()
    coord.join(threads)
