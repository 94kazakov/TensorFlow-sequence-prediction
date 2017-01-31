import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import data_help as dh


# load data
# find ids for randomizing data - TODO with shuffle=True flag from tf
# mask each batch - TODO with tf.train.batch(...pad..=True)
# how to avoid making placeholders of fixed size for time?
# TODO: add sigmoid activation to output product
# Make sure padding works (to ignore 0's during accuracy and loss count)
# Code Bayesian. 
# Abstract RNN into a separate class (make sure we can connect layer upon layer)
tf.reset_default_graph()
map_fn = tf.python.map_fn


# order of information from load_data
X = 0
XT = 1
Y = 2
YT = 3

# Parameters (options)
# max_length = maxlen
ops = {
            'epochs': 300, 
            'frame_size': 3,
            'n_hidden': 50,
            'n_classes': 50,
            'learning_rate': 0.001,
            'batch_size': 16,
            'max_length': 400
          }

# tf Graph input
seq_length = tf.placeholder(tf.int32)
x = tf.placeholder("float", [None, ops['max_length'], ops['frame_size']]) #None - for dynamic batch sizing
y = tf.placeholder("float", [None, ops['max_length'],  ops['n_classes']])

# Weights
W = {'out': tf.Variable(tf.random_normal([ops['n_hidden'], ops['n_classes']]))}
b = {'out': tf.Variable(tf.random_normal([ops['n_classes']]))}


def RNN(x, W, b):
    """
    Prepare data shape to match 'rnn' function req-s
    Current data input shape: (ops['max_length'], ops['batch_size'], frame_size)
    Required shape: 'ops['max_length']' tensors list of shape (ops['batch_size'], frame_size)
    """

    # Reshaping to (ops['max_length']*ops['batch_size'], frame_size)
    # x = tf.reshape(x, [-1, ops['frame_size']])
    # # Split to get a list of 'ops['max_length']' tensors of shape (ops['batch_size'], frame_size)
    # x = tf.split(0, ops['max_length'], x)
    
    # lstm cell
    lstm_cell = rnn_cell.BasicLSTMCell(ops['n_hidden'], forget_bias=1.0)
    
    # get lstm_cell's output
    # dynamic_rnn return by default: 
    #   outputs: [batch_size, max_time, cell.output_size]
      
    outputs, states = tf.nn.dynamic_rnn(
                                lstm_cell, 
                                x, 
                                dtype=tf.float32,
                                sequence_length=seq_length)
    
    #linear activation, using rnn innter loop last output
    output_projection = lambda x: tf.matmul(x, W['out']) + b['out']
    return map_fn(output_projection, outputs)


pred = RNN(x, W, b)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=ops['learning_rate']).minimize(cost)

# Evaluate the model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables
init = tf.global_variables_initializer()






train_set, valid_set, test_set = dh.load_data('data/reddit_test/reddit', sort_by_len=True)    
print "Loaded the set: train({}), valid({}), test({})".format(len(train_set),
                                                                len(valid_set),
                                                                  len(test_set))




# Sources:
# http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
# http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
# http://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
# https://danijar.com/variable-sequence-lengths-in-tensorflow/

# TODO what's up with batch_size - always 4?
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    epoch = 0
    while epoch < ops['epochs']:
        train_batch_indices = dh.get_minibatches_ids(len(train_set), ops['batch_size'], shuffle=True)
        epoch += 1
        for batch_indeces in train_batch_indices:
            #print batch_indeces
            # select examples from train set that correspond to each minibatch
            
            batch_x = [train_set[i][X] for i in batch_indeces]
            batch_xt = [train_set[i][XT] for i in batch_indeces]
            batch_y = [train_set[i][Y] for i in batch_indeces]
            batch_yt = [train_set[i][YT] for i in batch_indeces]
            # print np.array(batch_x).shape, batch_x
            # pad minibatch
            batch_x, batch_xt, batch_y, batch_yt, mask, batch_maxlen = dh.prepare_data(
                                                                            batch_x, 
                                                                            batch_xt, 
                                                                            batch_y, 
                                                                            batch_yt, 
                                                                            ops['max_length'], 
                                                                            ops['max_length'])
            # make an input set of dimensions (batch_size, max_length, frame_size)
            x_set = np.array([batch_x, batch_xt, batch_yt]).transpose([1,2,0])
            _, fetched_cost, fetched_accuracy = sess.run([optimizer, cost, accuracy], feed_dict={x: x_set, y: dh.embed_one_hot(batch_y, ops['n_classes'], ops['max_length']), seq_length: batch_maxlen})
            print fetched_cost, fetched_accuracy
