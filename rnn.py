import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import data_help as dh

# load data
# find ids for randomizing data - TODO with shuffle=True flag from tf
# mask each batch - TODO with tf.train.batch(...pad..=True)
# how to avoid making placeholders of fixed size for time?

tf.reset_default_graph()


# order of information from load_data
X = 0
XT = 1
Y = 2
YT = 3

# Parameters (options)
# n_steps = maxlen
ops = {
            'epochs': 3, 
            'frame_size': 3,
            'n_steps': 300,
            'n_hidden': 50,
            'n_classes': 50,
            'learning_rate': 0.001,
            'batch_size': 16,
            'max_length': 300
          }

# tf Graph input
x = tf.placeholder("float", [None, ops['max_length'], ops['frame_size']]) #None - for dynamic batch sizing
y = tf.placeholder("float", [None, ops['max_length'],  ops['n_classes']])

# Weights
W = {'out': tf.Variable(tf.random_normal([ops['n_hidden'], ops['n_classes']]))}
b = {'out': tf.Variable(tf.random_normal([ops['n_classes']]))}


def RNN(x, W, b):
    """
    Prepare data shape to match 'rnn' function req-s
    Current data input shape: (ops['n_steps'], ops['batch_size'], frame_size)
    Required shape: 'ops['n_steps']' tensors list of shape (ops['batch_size'], frame_size)
    """

    # Reshaping to (ops['n_steps']*ops['batch_size'], frame_size)
    # x = tf.reshape(x, [-1, ops['frame_size']])
    # # Split to get a list of 'ops['n_steps']' tensors of shape (ops['batch_size'], frame_size)
    # x = tf.split(0, ops['n_steps'], x)
    
    # lstm cell
    lstm_cell = rnn_cell.BasicLSTMCell(ops['n_hidden'], forget_bias=1.0)
    
    # get lstm_cell's output
    # dynamic_rnn return by default: 
    #   outputs: [batch_size, max_time, cell.output_size]
      
    outputs, states = tf.nn.dynamic_rnn(
                                lstm_cell, 
                                x, 
                                dtype=tf.float32,
                                sequence_length=dh.length(x))
    
    #linear activation, using rnn innter loop last output
    return [tf.matmul(outputs[i], W['out']) + b['out'] for i in range(ops['n_steps'])]


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
    train_batch_indices = dh.get_minibatches_ids(len(train_set[X]), ops['batch_size'], shuffle=True)
    while epoch < ops['epochs']:
        for batch_indeces in train_batch_indices:
            # select examples from train set that correspond to each minibatch
            
            batch_x = [train_set[i][X] for i in batch_indeces]
            batch_xt = [train_set[i][XT] for i in batch_indeces]
            batch_y = [train_set[i][Y] for i in batch_indeces]
            batch_yt = [train_set[i][YT] for i in batch_indeces]
            print np.array(batch_x).shape, batch_x
            # pad minibatch
            batch_x, batch_xt, batch_y, batch_yt, mask = dh.prepare_data(batch_x, batch_xt, batch_y, batch_yt, maxlen=-1)
            # make an input set of dimensions (batch_size, n_steps, frame_size)
            x_set = np.array([batch_x, batch_xt, batch_yt]).transpose([1,2,0])
            sess.run(optimizer, feed_dict={x: x_set, y: dh.embed_one_hot(batch_y, ops['n_classes'], ops['max_length'])})

