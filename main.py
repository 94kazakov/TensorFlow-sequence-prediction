import tensorflow as tf
import numpy as np
import data_help as DH
import tensor_classes_helpers as TCH
import sys
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


ops = {
            'epochs': 500,
            'frame_size': 3,
            'n_hidden': 50,
            'n_classes': 50, # aka n_input
            'learning_rate': 0.0005,
            'batch_size': 64,
            'max_length': 100,
            'encoder': 'CTGRU_5_1',
            'dataset': 'data/synthetic_cluster/cluster3',
            'overwrite': False,
            "write_history": True, #whether to write the history of training
            'model_save_name': None,
            'model_load_name': None,
            'store_graph': False,
            'collect_histograms': False,
            'unique_mus_alphas': False, #HPM only
            '1-to-1': True, #HPM only - forces it to be
            'embedding': False, #only for CTGRU so far TODO: extract to be generic
            'embedding_size': 30,
            'vocab_size': 10000,
            'task': "CLASS" #CLASS vs PRED
          }

# load the dataset
train_set, valid_set, test_set = DH.load_data(ops['dataset'], sort_by_len=True)
if ops['embedding']:
    extract_ids = lambda set: np.concatenate(np.array([set[i][0] for i in range(len(set))]))
    all_ids = np.concatenate([np.array(extract_ids(train_set)),
                             np.array(extract_ids(valid_set)),
                             np.array(extract_ids(test_set))])


    count, dictionary, reverse_dictionary = DH.build_dataset(all_ids, ops['vocab_size'])
    print "Popular ids:", count[:5]
    train_set = DH.remap_data(train_set, dictionary)
    valid_set = DH.remap_data(valid_set, dictionary)
    test_set = DH.remap_data(test_set, dictionary)

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

if ops['embedding']:
    P_y = tf.placeholder("int32",
                        [None, ops['max_length']], name='y')
else:
    P_y = TCH.output_placeholder(max_length_seq=ops['max_length'],
                            number_of_classes=ops['n_classes'], name='y')
P_mask = tf.placeholder("float", 
                        [None, ops['max_length']], name='mask')
P_batch_size = tf.placeholder("float", None)

T_embedding_matrix = None
if ops['embedding']:
    T_embedding_matrix = tf.Variable(
        tf.random_uniform([ops['vocab_size'], ops['embedding_size']], -1.0, 1.0))



print "MODE: ", ops['encoder']
print "Store history of learning:", ops['write_history']


# params init
if ops['encoder'] == "LSTM":
    params_lstm = TCH.LSTM_params_init(ops)
elif ops['encoder'] == "HPM":
    params_hpm = TCH.HPM_params_init(ops)
elif ops['encoder'] == "LSTM_RAW":
    params_lstm = TCH.LSTM_raw_params_init(ops)
elif ops['encoder'] == "GRU":
    params_gru = TCH.GRU_params_init(ops)
elif ops['encoder'] == "CTGRU_5_1" or ops['encoder'] == "CTGRU_5_3":
    params_ctgru = TCH.CTGRU_params_init(ops)

# predict using encoder
if ops['encoder'] == 'LSTM':
    T_pred, T_summary_weights, debugging_stuff = TCH.RNN([P_x, P_len], ops, params_lstm)
    T_pred = tf.transpose(T_pred, [1,0,2])
elif ops['encoder'] == 'HPM':
    T_pred, T_summary_weights, debugging_stuff = TCH.HPM(P_x, P_len, P_batch_size, ops, params_hpm, P_batch_size)
elif ops['encoder'] == "LSTM_RAW":
    T_pred, T_summary_weights, debugging_stuff = TCH.LSTM([P_x, P_len, P_batch_size], ops, params_lstm)
elif ops['encoder'] == "GRU":
    T_pred, T_summary_weights, debugging_stuff = TCH.GRU([P_x, P_len, P_batch_size], ops, params_gru)
elif ops['encoder'] == "CTGRU_5_3":
    T_pred, T_summary_weights, debugging_stuff = TCH.CTGRU_5_3([P_x, P_len, P_batch_size, T_embedding_matrix], ops, params_ctgru)
elif ops['encoder'] == "CTGRU_5_1":
    T_pred, T_summary_weights, debugging_stuff = TCH.CTGRU_5_1([P_x, P_len, P_batch_size, T_embedding_matrix], ops, params_ctgru)


# (mean (batch_size):
#   reduce_sum(n_steps):
#       P_mask * (-reduce_sum(classes)):
#           truth * predicted_distribution)
if ops['embedding']: #TODO: cos distance okay? since we only care about directions anyway
    y_answer = tf.nn.embedding_lookup(T_embedding_matrix, P_y)
    T_cost = tf.reduce_sum(
        tf.reduce_sum(
            tf.reduce_sum(
               tf.abs((y_answer*T_pred)/(tf.norm(y_answer,2,keep_dims=True)*tf.norm(T_pred,2,keep_dims=True)) - 1.0)**2,
                reduction_indices=[2]) * P_mask,
            reduction_indices=[1])) / tf.reduce_sum(tf.reduce_sum(P_mask))
else:
    if ops['task'] == 'PRED':
        y_answer = P_y
        T_cost = tf.reduce_sum(
                    tf.reduce_sum(
                        - tf.reduce_sum(
                            (y_answer * tf.log(T_pred)),
                        reduction_indices=[2]) * P_mask,
                    reduction_indices=[1])) / tf.reduce_sum(tf.reduce_sum(P_mask))

        # Evaluate the model
        T_correct_pred = tf.cast(tf.equal(tf.argmax(T_pred, 2), tf.argmax(y_answer, 2)), tf.float32) * P_mask
        T_accuracy = tf.reduce_sum(tf.reduce_sum(tf.cast(T_correct_pred, tf.float32))) / tf.reduce_sum(
            tf.reduce_sum(P_mask))
    elif ops['task'] == "CLASS":
        y_answer = P_y
        T_cost = tf.reduce_sum(
                - tf.reduce_sum(
                    (y_answer * tf.log(T_pred)),
                    reduction_indices=[2])[:,-1]
                ) / tf.reduce_sum(tf.reduce_sum(y_answer))

        # Evaluate the model
        T_correct_pred = tf.cast(tf.equal(tf.argmax(T_pred, 2), tf.argmax(y_answer, 2)), tf.float32)
        T_accuracy = tf.reduce_sum(tf.cast(T_correct_pred[:,-1], tf.float32)) / tf.reduce_sum(
            tf.reduce_sum(y_answer))

T_optimizer = tf.train.AdamOptimizer(learning_rate=ops['learning_rate']).minimize(T_cost)


# Initialize the variables
init = tf.global_variables_initializer()






if ops['store_graph']:
    # Model parameters
    logs_path = '/Users/denis/Documents/hawkes/logs/run1'
    # Empty path if nonempty:
    DH.empty_directory(logs_path)
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # This just didn't work for me.
    # T_sess = tf_debug.LocalCLIDebugWrapperSession(T_sess)
    # T_sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
saver = tf.train.Saver()
try:
    new_saver = tf.train.import_meta_graph('saved_models/' + ops['model_load_name'] + '.meta')
    new_saver.restore(T_sess, 'saved_models/' + ops['model_load_name'])
    print "Model Loaded from " + ops['model_load_name']
except:
    print "Failed to load the model: " + str(ops['model_load_name'])
    T_sess.run(init)





epoch = 0
counter = 0
summary, deb_var, summary_weights, y_answer = None, None, None, None

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

        if ops['embedding']:
            # batch_y = (batch, n_steps_padded)
            y_answer = np.array(batch_y).astype(np.int32)
        else:
            # (batch_size, steps, n_classes)
            y_answer = DH.embed_one_hot(batch_y, 0.0, ops['n_classes'], ops['max_length'])
        _, deb_var, summary_weights = T_sess.run(
                                                [T_optimizer, debugging_stuff, T_summary_weights],
                                                feed_dict={
                                                            P_x: x_set,
                                                            P_y: y_answer,
                                                            P_len: batch_maxlen,
                                                            P_mask: mask,
                                                            P_batch_size: batch_size})

        print counter
        names = ["h","o", "h_prev","o_prev","q","s","sigma","r","rho",'mul','decay']
        np.set_printoptions(precision=5)
        for i,var in enumerate(deb_var):
            var = np.array(var)
            # if names[i] in ['o_prev','h_prev','q','h_hat']:
            #     print '\n'
            #     if (var < -1.0).any():
            #         print names[i], var[var < -1.0]
            #     elif (var > 1.0).any():
            #         print names[i], var[var > 1.0]
            # if names[i] in ['sigma', 'rho']:
            #     print '\n'
            #     not_sum_to_one = lambda x: np.abs(np.sum(x, axis=3) - 1.0) > 0.00000001
            #     if (not_sum_to_one(var)).any():
            #         print names[i], np.sum(var[not_sum_to_one(var)], axis=1) - 1.0
            #     #print names[i], list(var[:,0,:])#200,64,50
            if np.isnan(var).any():
                #import pdb; pdb.set_trace()
                print "FOUND NAN"#, names[i]
                sys.exit()

        #Print parameters
        # for v in tf.global_variables():
        #     v_ = T_sess.run(v)
        #     print v.name


        if ops['collect_histograms']:
            writer.add_summary(summary_weights, counter)
    # print "alphas:", T_sess.run(tf.Print(params['alpha'], [params['alpha']]))

    # Evaluating model at each epoch
    datasets = [train_set, test_set, valid_set]
    dataset_names = ['train', 'test', 'valid']

    accuracy_entry, losses_entry = TCH.errors_and_losses(T_sess, P_x, P_y,
                                                        P_len, P_mask, P_batch_size, T_accuracy, T_cost, T_embedding_matrix,
                                                        dataset_names, datasets, ops)
    print "Epoch:{}, Accuracy:{}, Losses:{}".format(epoch, accuracy_entry, losses_entry)

    if ops['model_save_name'] != None:
        print "Model Saved as ", ops['model_save_name']
        saver.save(T_sess, 'saved_models/' + ops['model_save_name'])

    if ops['write_history']:
        DH.write_history(accuracy_entry, 'records/acc.txt', epoch, ops['overwrite'])
        DH.write_history(losses_entry, 'records/loss.txt', epoch, ops['overwrite'])

        






