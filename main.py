import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import data_help as DH
import tensor_classes_helpers as TCH


"""
STRUCTURE:
data_help:
    + read_file_time_sequences(fname)
    + load_data(dir, sort_by_len=True, valid_ratio=0.1)
    + get_minibatches_ids(n, minibatch_size, shuffle=True)
    + prepare_data(ox, oxt, oy, oyt, maxlen=None, extended_len=0)
    + embed_one_hot(batch_array, depth, length)
    + length(sequence)
main:
    Main logic is here as well as all of the manipulations.
    We store the variables for tensorflow only here. 
    - parameters:
        RNN/HPM (decide here)
        number of epochs
        dimensions, hidden layers
    - init layers
    - init solvers
    - all manipulations
tensor_classes_helpers:
    Here, we have functions of encoders and functions to
    manipulate tensorflow variables
    RNN:
        + init(input_dim, output_dim)
    HPM:
    + encoders
    + weights, bias inits


"""

# load data
# find ids for randomizing data - TODO with shuffle=True flag from tf
# mask each batch - TODO with tf.train.batch(...pad..=True)
# how to avoid making placeholders of fixed size for time?
# TODO: add sigmoid activation to output product
# Make sure padding works (to ignore 0's during accuracy and loss count)
# Code Bayesian. 
# Abstract RNN into a separate class (make sure we can connect layer upon layer)
# extend weights/bias inits to support differet init distributions
# accuracy on train, test, val sets

# Sources:
# http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
# http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
# http://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
# https://danijar.com/variable-sequence-lengths-in-tensorflow/

# order of information from load_data
X = 0
XT = 1
Y = 2
YT = 3
# Model parameters
ops = {
            'epochs': 300, 
            'frame_size': 3,
            'n_hidden': 50,
            'n_classes': 50,
            'learning_rate': 0.001,
            'batch_size': 64,
            'max_length': 400,
            'encoder': 'LSTM',
            'dataset': 'data/reddit/reddit'
          }

# load the dataset
train_set, valid_set, test_set = DH.load_data(ops['dataset'], sort_by_len=True)    
print "Loaded the set: train({}), valid({}), test({})".format(len(train_set),
                                                                len(valid_set),
                                                                  len(test_set))

# Restart the graph
tf.reset_default_graph()
# Graph placeholders
seq_length = tf.placeholder(tf.int32)
x = TCH.input_placeholder(max_length_seq=ops['max_length'], 
                            frame_size=ops['frame_size'])

y = TCH.output_placeholder(max_length_seq=ops['max_length'], 
                            number_of_classes=ops['n_classes'])
# Graph weights
W = {'out': TCH.weights_init(n_input=ops['n_hidden'], 
                                n_output=ops['n_classes'])}
b = {'out': TCH.bias_init(
                    ops['n_classes'])}


# predict using encoder
if ops['encoder'] == 'LSTM':
    pred = TCH.RNN(x, W['out'], b['out'], seq_length, ops['n_hidden'])
else:
    print "Not yet"
# Loss and optimizer (automatically updates all of the weights)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=ops['learning_rate']).minimize(cost) 

# Evaluate the model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initialize the variables
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    epoch = 0
    while epoch < ops['epochs']:
        train_batch_indices = DH.get_minibatches_ids(len(train_set), ops['batch_size'], shuffle=True)
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
            batch_x, batch_xt, batch_y, batch_yt, mask, batch_maxlen = DH.prepare_data(
                                                                            batch_x, 
                                                                            batch_xt, 
                                                                            batch_y, 
                                                                            batch_yt, 
                                                                            maxlen=ops['max_length'], 
                                                                            extended_len=ops['max_length'])
            # make an input set of dimensions (batch_size, max_length, frame_size)
            x_set = np.array([batch_x, batch_xt, batch_yt]).transpose([1,2,0])
            _, fetched_cost, fetched_accuracy = sess.run(
                                                    [optimizer, cost, accuracy], 
                                                    feed_dict={
                                                                x: x_set, 
                                                                y: DH.embed_one_hot(batch_y, ops['n_classes'], ops['max_length']), 
                                                                seq_length: batch_maxlen})

            print "Loss: {}, \tAcc:{}".format(fetched_cost, fetched_accuracy)
