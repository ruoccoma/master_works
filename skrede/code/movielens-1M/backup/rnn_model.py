import tensorflow as tf
import pickle

location  = '/home/ole/recSys-pro/movielens-1M/'
n_hidden  = 100

# Automatically load some other parameters specified in preprocessing and elsewhere
meta_pickle = location + 'meta.pickle'
meta_data   = pickle.load( open(meta_pickle, 'rb'))
max_sequence_length = meta_data['max_sequence_length']
k                   = meta_data['k']
train_files         = meta_data['train_files']
n_classes           = meta_data['n_classes']
batch_size          = meta_data['batch_size']
n_readers            = len(train_files)
n_preprocess_threads = n_readers*2
pickle.dump(meta_data, open(meta_pickle, 'wb'))

# Create library of 1-hots
#tmp = [i for i in range(n_classes)]
#one_hots = tf.one_hot(tmp, depth=n_classes)


#############
# READ DATA #
#############

# Create filename_queue
filename_queue = tf.train.string_input_producer(train_files, shuffle=True)

min_after_dequeue = 1024
capacity          = min_after_dequeue + 3*batch_size
examples_queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        dtypes=[tf.string])

# Create multiple readers to populate the queue of examples
enqueue_ops = []
for i in range(n_readers):
    reader = tf.TextLineReader()
    _key, value = reader.read(filename_queue)
    enqueue_ops.append(examples_queue.enqueue([value]))

tf.train.queue_runner.add_queue_runner(
        tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
example_string = examples_queue.dequeue()

# Default values, and type of the columns, first is sequence_length
# +1 since first field is sequence length
record_defaults = [[0]]*(max_sequence_length+1)

parsed_examples = []
for thread_id in range(n_preprocess_threads):
    example = tf.decode_csv(value, record_defaults=record_defaults)

    # Create 1-HOT vectors of the sequence
    one_hots = example[1:]
    one_hots = tf.reshape(one_hots, [-1])
    one_hots = tf.one_hot(one_hots, depth=n_classes)

    # Split the row into input/target values
    sequence_length = example[0]
    features = one_hots[:-1]
    targets  = one_hots[1:]
    #features = example[1:-1]
    #targets  = example[2:]
    #features = tf.nn.embedding_lookup(one_hots, features)
    #targets  = tf.nn.embedding_lookup(one_hots, targets)

    parsed_examples.append([sequence_length, features, targets])

# Batch together examples
session_length, x, y = tf.train.batch_join(
        parsed_examples, 
        batch_size=batch_size,
        capacity=2*n_preprocess_threads*batch_size)


# Parse the examples in a batch

###############
# RNN NETWORK #
###############

output, state = tf.nn.dynamic_rnn(
        tf.nn.rnn_cell.GRUCell(n_hidden),
        x,
        dtype=tf.float32,
        sequence_length=session_length)
            
layer = {'weights':tf.Variable(tf.random_normal([n_hidden, n_classes])),
         'biases':tf.Variable(tf.random_normal([n_classes]))}

# Flatten to apply same weights to all time steps.
output = tf.reshape(output, [-1, n_hidden], name="flatOuput")

# Add dropout
keep_prob = tf.placeholder(tf.float32)
output = tf.nn.dropout(output, keep_prob)

# The models prediction
prediction = tf.matmul(output, layer['weights'], name="prediction")# + layer['biases'] TODO

# Reduce sum, since average divides by max_length, which is often wrong
error = tf.nn.softmax_cross_entropy_with_logits(prediction, y, name="crossEntropy")
cost  = tf.reduce_sum(error, name="cost")

# The training 'function'
optimizer = tf.train.AdamOptimizer(name="optimizer").minimize(cost)

# Top k predictions for testing
top_pred_vals, top_pred_indices = tf.nn.top_k(prediction, k=k, sorted=True, name="topPredictions")

