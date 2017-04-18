import tensorflow as tf
import numpy as np
import data_help as DH
import tensor_classes_helpers as TCH
from tensorflow.python import debug as tf_debug
import os

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

# Make sure padding works (to ignore 0's during accuracy and loss count)
# Right now placeholders are length size (400) and I just specify what's the max lengths of sequences using T_l into the LSTM cell
# See distributions of weights over time & their activations
# TODO: how does it learn to 40% on small set. Doest it generalize?
# TODO: HPM's representation is not distributed. It's direct (each event is a neuron). Unlike LSTM.
#       Trying embedding in this case should help the issue I think.
# Sources:
# http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
# http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
# http://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
# https://danijar.com/variable-sequence-lengths-in-tensorflow/
# debugging: https://wookayin.github.io/tensorflow-talk-debugging/#40


# Model parameters
logs_path = '/Users/denis/Documents/hawkes/logs/run1'
# Empty path if nonempty:
DH.empty_directory(logs_path)



ops = {
            'epochs': 500,
            'frame_size': 3,
            'n_hidden': 50,
            'n_classes': 50,
            'learning_rate': 0.0005,
            'batch_size': 64,
            'max_length': 200,
            'encoder': 'HPM',
            'dataset': 'data/reddit/reddit',
            'overwrite': False,
            'model_save_name': "HPM_mu_alpha_scaled_april18",
            'model_load_name': None,
            'debug_tensorflow': False,
            'collect_histograms': False
          }

# load the dataset
train_set, valid_set, test_set = DH.load_data(ops['dataset'], sort_by_len=True)    
print "Loaded the set: train({}), valid({}), test({})".format(len(train_set),
                                                                len(valid_set),
                                                                  len(test_set))

# Restart the graph
tf.reset_default_graph()
T_sess = tf.Session()

# Graph placeholders
P_len = tf.placeholder(tf.int32)
P_x = TCH.input_placeholder(max_length_seq=ops['max_length'], 
                            frame_size=ops['frame_size'], name="x")

P_y = TCH.output_placeholder(max_length_seq=ops['max_length'], 
                            number_of_classes=ops['n_classes'], name='y')
P_mask = tf.placeholder("float", 
                        [None, ops['max_length']], name='mask')
P_batch_size = tf.placeholder("float", None)

# Graph weights
W = {'out': TCH.weights_init(n_input=ops['n_hidden'], 
                                n_output=ops['n_classes'],
                                name='W_out')}
b = {'out': TCH.bias_init(
                    ops['n_classes'],
                    name='b_out')}

# params init
# predict using encoder
if ops['encoder'] == 'LSTM':
    params = None
else:
    params = TCH.HPM_params_init(ops)

# predict using encoder
if ops['encoder'] == 'LSTM':
    T_pred = tf.transpose(TCH.RNN(P_x, W['out'], b['out'], P_len, ops['n_hidden'], ops), [1,0,2])
else:
    if ops['collect_histograms']:
        T_pred, T_summary_weights, debugging_stuff = TCH.HPM(P_x, ops, params, P_batch_size)
    else:
        T_pred, debugging_stuff = TCH.HPM(P_x, ops, params, P_batch_size)

    #T_pred, debugging_val = TCH.HPM(P_x, ops, params, P_batch_size)
    #T_pred = tf.transpose(T_pred, [1,0,2])

# for y = [None, n_classes]:
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# (mean (batch_size):
#   reduce_sum(n_steps):
#       P_mask * (-reduce_sum(classes)):
#           truth * predicted_distribution)
T_cost = tf.reduce_sum(
            tf.reduce_sum(
                - tf.reduce_sum(
                    (P_y * tf.log(T_pred)),
                reduction_indices=[2]) * P_mask,
            reduction_indices=[1])) / tf.reduce_sum(tf.reduce_sum(P_mask))

T_optimizer = tf.train.AdamOptimizer(learning_rate=ops['learning_rate']).minimize(T_cost) 

# Evaluate the model
T_correct_pred = tf.cast(tf.equal(tf.argmax(T_pred,2), tf.argmax(P_y,2)), tf.float32) * P_mask
# T_correct_pred = tf.equal(tf.argmax(T_pred,2), tf.argmax(T_pred,2))
T_accuracy = tf.reduce_sum(tf.reduce_sum(tf.cast(T_correct_pred, tf.float32)))/tf.reduce_sum(tf.reduce_sum(P_mask))

tf.summary.scalar('T_cost', T_cost)
tf.summary.scalar('accuracy', T_accuracy)
T_summary_op = tf.summary.merge_all()

# Initialize the variables
init = tf.global_variables_initializer()






if ops['debug_tensorflow']:
    T_sess = tf_debug.LocalCLIDebugWrapperSession(T_sess)
    T_sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

saver = tf.train.Saver()
try:
    new_saver = tf.train.import_meta_graph('saved_models/' + ops['model_load_name'] + '.meta')
    new_saver.restore(T_sess, 'saved_models/' + ops['model_load_name'])
    print "Model Loaded from " + ops['model_load_name']
except:
    print "Failed to load the model: " + str(ops['model_load_name'])
    T_sess.run(init)



writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

epoch = 0
counter = 0
summary, deb_var, summary_weights = None, None, None

print "Format: Train, Test, Valid"
while epoch < ops['epochs']:
    train_batch_indices = DH.get_minibatches_ids(len(train_set), ops['batch_size'], shuffle=True)
    epoch += 1
    for batch_indeces in train_batch_indices:
        counter += 1
        x_set, batch_y, batch_maxlen, batch_size, mask = DH.pick_batch(
                                            dataset = train_set,
                                            batch_indeces = batch_indeces,
                                            max_length = ops['max_length'])
        # x_set: [batch_size, max_length, frame_size]

        tensor_list_to_recover = None
        if ops['collect_histograms']:
            tensor_list_to_recover = [T_optimizer, debugging_stuff, T_summary_weights]
        else:
            tensor_list_to_recover = [T_optimizer, debugging_stuff]

        recovered_variables = T_sess.run(
                                                tensor_list_to_recover,
                                                feed_dict={
                                                            P_x: x_set,
                                                            P_y: DH.embed_one_hot(batch_y, 0.0, ops['n_classes'], ops['max_length']),
                                                            P_len: batch_maxlen,
                                                            P_mask: mask,
                                                            P_batch_size: batch_size})

        if ops['collect_histograms']:
            _, deb_var, summary_weights = recovered_variables
        else:
            _, deb_var = recovered_variables

        # Print parameters

        # for v in tf.global_variables():
        #     v_ = T_sess.run(v)
        #     print v.name
        #     print v_
        #     print '\n'
        #import pdb; pdb.set_trace()
        #T_sess.run(tf.Print(TCH.cut_up_x(x_set, ops)[0], [x_set[0], TCH.cut_up_x(x_set, ops)[0]]))
        #extract_batch = lambda arr, batch_i, n_hid_i: np.transpose(np.array([arr[i][batch_i][n_hid_i] for i in range(arr.shape[0])]))
        #print extract_batch(deb_var, 0, 0)
        #print extract_batch(deb_var, 0, 1)
        #print extract_batch(deb_var, 0, 2)
        #print pred
        # T_sess.run(
        #     [TCH._step],
        #     feed_dict={
        #         P_x: [[[1, 1, 2, 3, 1, 4, 1, 1, 1],
        #                [0.00000, 0.94681, 35.00047, 2.82917, 10.17781, 1.39126, 5.88494, 9.74331, 3.19285],
        #                [1, 2, 3, 1, 4, 1, 1, 1, 1]]],
        #         P_y: DH.embed_one_hot(
        #             [0.94681, 35.00047, 2.82917, 10.17781, 1.39126, 5.88494, 9.74331, 3.19285, 21.74557], 0.0,
        #             ops['n_classes'], ops['max_length']),
        #         P_len: 9,
        #         P_mask: [[1, 1, 1, 1, 1, 1, 1, 1, 1]],
        #         P_batch_size: 1})

        if ops['collect_histograms']:
            writer.add_summary(summary_weights, counter)
    # print "alphas:", T_sess.run(tf.Print(params['alpha'], [params['alpha']]))
    # Evaluating model at each epoch
    datasets = [train_set, test_set, valid_set]
    dataset_names = ['train', 'test', 'valid']

    accuracy_entry, losses_entry = TCH.errors_and_losses(T_sess, P_x, P_y,
                                                        P_len, P_mask, P_batch_size, T_accuracy, T_cost,
                                                        dataset_names, datasets, ops)
    print "Epoch:{}, Accuracy:{}, Losses:{}".format(epoch, accuracy_entry, losses_entry)

    if ops['model_save_name'] != None:
        print "Model Saved as ", ops['model_save_name']
        saver.save(T_sess, 'saved_models/' + ops['model_save_name'])
    DH.write_history(accuracy_entry, 'records/acc.txt', epoch, ops['overwrite'])
    DH.write_history(losses_entry, 'records/loss.txt', epoch, ops['overwrite'])

        






