import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import data_help as DH



def input_placeholder(max_length_seq=100, 
                        frame_size=3):
    
    x = tf.placeholder("float", 
                        [None, max_length_seq, 
                        frame_size]) #None - for dynamic batch sizing
    return x

def output_placeholder(max_length_seq=100, 
                        number_of_classes=50):
    
    y = tf.placeholder("float", 
                        [None, max_length_seq,  
                        number_of_classes])
    return y

def weights_init(n_input, n_output):
    W = tf.Variable(tf.random_normal([n_input, n_output]))
    return W

def bias_init(n_output):
    b = tf.Variable(tf.random_normal([n_output]))
    return b

def errors_and_losses(sess, P_x, P_y, P_len, P_mask, T_accuracy,  T_cost, dataset_names, datasets, ops):
    # passes all needed tensor placeholders to fill with passed datasets
    # computers errors and losses for train/test/validation sets
    # Depending on what T_accuracy, T_cost are, different nets can be called
    accuracy_entry = []
    losses_entry = []
    for i in range(len(dataset_names)):
        dataset = datasets[i]
        dataset_name = dataset_names[i]
        batch_indeces_arr = DH.get_minibatches_ids(len(dataset), ops['batch_size'], shuffle=True)

        acc_avg = 0.0
        loss_avg = 0.0
        for batch_ids in batch_indeces_arr:
            x_set, batch_y, batch_maxlen, mask = DH.pick_batch(
                                            dataset = dataset,
                                            batch_indeces = batch_ids, 
                                            max_length = ops['max_length']) 
            print batch_maxlen
            accuracy_batch, cost_batch = sess.run([T_accuracy, T_cost],
                                                    feed_dict={
                                                        P_x: x_set, 
                                                        P_y: DH.embed_one_hot(batch_y, ops['n_classes'], ops['max_length']), 
                                                        P_len: batch_maxlen,
                                                        P_mask: mask})
            acc_avg += accuracy_batch
            loss_avg += cost_batch
        accuracy_entry.append(acc_avg/len(batch_indeces_arr))
        losses_entry.append(cost_batch/len(batch_indeces_arr))
    return accuracy_entry, losses_entry
    

def RNN(x, T_W, T_b, T_seq_length, n_hidden):
    # lstm cell
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    
    # get lstm_cell's output
    # dynamic_rnn return by default: 
    #   outputs: [batch_size, max_time, cell.output_size]
      
    outputs, states = tf.nn.dynamic_rnn(
                                lstm_cell, 
                                x, 
                                dtype=tf.float32,
                                sequence_length=T_seq_length)
    
    # linear activation, using rnn innter loop last output
    # project into class space: x-[max_time, hidden_units], T_W-[hidden_units, n_classes]
    output_projection = lambda x: tf.nn.softmax(tf.matmul(x, T_W) + T_b)

    return tf.python.map_fn(output_projection, outputs)




