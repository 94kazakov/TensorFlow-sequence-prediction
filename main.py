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

# # find ids for randomizing data - TODO with shuffle=True flag from tf
# # mask each batch - TODO with tf.train.batch(...pad..=True)
# Make sure padding works (to ignore 0's during accuracy and loss count)
# Right now placeholders are length size (400) and I just specify what's the max lengths of sequences using T_l into the LSTM cell
# See distributions of weights over time & their activations

# Sources:
# http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
# http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
# http://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
# https://danijar.com/variable-sequence-lengths-in-tensorflow/


# Model parameters
logs_path = '/Users/denis/Documents/hawkes/logs/run1'
ops = {
            'epochs': 300, 
            'frame_size': 3,
            'n_hidden': 50,
            'n_classes': 50,
            'learning_rate': 0.001,
            'batch_size': 64,
            'max_length': 400,
            'encoder': 'LSTM',
            'dataset': 'data/reddit_test/reddit',
            'overwrite': True
          }

# load the dataset
train_set, valid_set, test_set = DH.load_data(ops['dataset'], sort_by_len=True)    
print "Loaded the set: train({}), valid({}), test({})".format(len(train_set),
                                                                len(valid_set),
                                                                  len(test_set))

# Restart the graph
tf.reset_default_graph()
# Graph placeholders
P_len = tf.placeholder(tf.int32)
P_x = TCH.input_placeholder(max_length_seq=ops['max_length'], 
                            frame_size=ops['frame_size'])

P_y = TCH.output_placeholder(max_length_seq=ops['max_length'], 
                            number_of_classes=ops['n_classes'])
# Graph weights
W = {'out': TCH.weights_init(n_input=ops['n_hidden'], 
                                n_output=ops['n_classes'])}
b = {'out': TCH.bias_init(
                    ops['n_classes'])}


# predict using encoder
if ops['encoder'] == 'LSTM':
    T_pred = TCH.RNN(P_x, W['out'], b['out'], P_len, ops['n_hidden'])
else:
    print "Not yet"
# Loss and optimizer (automatically updates all of the weights)
# want to use mask to disregard information from padded data
T_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(T_pred, P_y))
T_optimizer = tf.train.AdamOptimizer(learning_rate=ops['learning_rate']).minimize(T_cost) 

# Evaluate the model
T_correct_pred = tf.equal(tf.argmax(T_pred,2), tf.argmax(P_y,2))
T_accuracy = tf.reduce_mean(tf.cast(T_correct_pred, tf.float32))

tf.summary.scalar('T_cost', T_cost)
tf.summary.scalar('accuracy', T_accuracy)
T_summary_op = tf.summary.merge_all()

# Initialize the variables
init = tf.global_variables_initializer()



with tf.Session() as T_sess:
    T_sess.run(init)
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    epoch = 0
    counter = 0
    while epoch < ops['epochs']:
        train_batch_indices = DH.get_minibatches_ids(len(train_set), ops['batch_size'], shuffle=True)
        epoch += 1
        for batch_indeces in train_batch_indices:
            counter += 1

            x_set, batch_y, batch_maxlen, mask = DH.pick_batch(
                                                dataset = train_set,
                                                batch_indeces = batch_indeces, 
                                                max_length = ops['max_length'])            
            _, summary = T_sess.run(
                                                    [T_optimizer, T_summary_op], 
                                                    feed_dict={
                                                                P_x: x_set, 
                                                                P_y: DH.embed_one_hot(batch_y, ops['n_classes'], ops['max_length']), 
                                                                P_len: batch_maxlen})


            # writer.add_summary(summary, counter)

        # Evaluating model at each epoch
        datasets = [train_set, test_set, valid_set]
        dataset_names = ['train', 'test', 'valid']
        
        accuracy_entry, losses_entry = TCH.errors_and_losses(T_sess, P_x, P_y, 
                                                            P_len, T_accuracy, T_cost, 
                                                            dataset_names, datasets, ops)
        print accuracy_entry, losses_entry
        DH.write_history(accuracy_entry, 'records/acc.txt', epoch, ops['overwrite'])
        DH.write_history(losses_entry, 'records/loss.txt', epoch, ops['overwrite'])
        
        






