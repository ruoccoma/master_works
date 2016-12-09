import tensorflow as tf
import pickle

location  = '/home/ole/recSys-pro/yoochoose/'
n_hidden  = 100

# Automatically load some other parameters specified in preprocessing and elsewhere
meta_pickle = location + 'meta.pickle'
meta_data   = pickle.load( open(meta_pickle, 'rb'))
max_sequence_length = meta_data['max_sequence_length']
k                   = meta_data['k']
train_files         = meta_data['train_files']
n_classes           = meta_data['n_classes']
batch_size          = meta_data['batch_size']
max_epochs          = meta_data['max_epochs']
n_readers            = 8
n_preprocess_threads = 8
pickle.dump(meta_data, open(meta_pickle, 'wb'))


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

    # Split the row into input/target values
    sequence_length = example[0]
    features = example[1:-1]
    targets  = example[2:]
    
    # Create a list with the value of sequence_length. E.g. 5 -> [5]
    input_length = tf.expand_dims(sequence_length, 0)
    # Use this to mask out outputs from the rnn, so we don't train on padding
    indicator = tf.reshape(tf.sequence_mask(input_length, maxlen=max_sequence_length-1, dtype=tf.int32), [-1])

    parsed_examples.append([sequence_length, features, targets, indicator])

# Batch together examples
session_length, x_unparsed, y_unparsed, mask = tf.train.batch_join(
        parsed_examples, 
        batch_size=batch_size,
        capacity=2*n_preprocess_threads*batch_size,
        dynamic_pad=False)


# Parse the examples in a batch
x = tf.one_hot(x_unparsed, depth=n_classes)
#y = tf.one_hot(y_unparsed, depth=n_classes)
y = tf.cast(y_unparsed, tf.int32)


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

# Reshape the prediction to fit in the loss function
prediction_reshaped = tf.reshape(prediction, [batch_size, max_sequence_length-1, n_classes])

# Compute losses
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(prediction_reshaped, y, name="losses")

# Use the masks to filter out loss on padded values that appears in shorter sequences
mask_weights = tf.to_float(mask)
batch_loss = tf.div(tf.reduce_sum(tf.mul(losses, mask_weights)),
        tf.reduce_sum(mask_weights),
        name="batch_loss")

# The optimizer for training
optimizer = tf.train.AdamOptimizer(name="optimizer").minimize(batch_loss)

# Top k predictions for testing
top_pred_vals, top_pred_indices = tf.nn.top_k(prediction, k=k, sorted=True, name="topPredictions")

top_k_preds = tf.reshape(top_pred_indices, [batch_size, max_sequence_length-1, k])
