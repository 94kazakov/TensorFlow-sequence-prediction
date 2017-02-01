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

def RNN(x, W_tensor, b_tensor, seq_length_tensor, n_hidden):
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
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    
    # get lstm_cell's output
    # dynamic_rnn return by default: 
    #   outputs: [batch_size, max_time, cell.output_size]
      
    outputs, states = tf.nn.dynamic_rnn(
                                lstm_cell, 
                                x, 
                                dtype=tf.float32,
                                sequence_length=seq_length_tensor)
    
    #linear activation, using rnn innter loop last output
    output_projection = lambda x: tf.nn.relu(tf.matmul(x, W_tensor) + b_tensor)
    return tf.python.map_fn(output_projection, outputs)




