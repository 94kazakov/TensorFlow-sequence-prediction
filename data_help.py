import tensorflow as tf
import numpy as np
import random

"""
# TODO:
# 1) save models: https://www.tensorflow.org/how_tos/variables/
# 2) abstract away cell type, data-outputs, solver. more structure:
# http://danijar.com/structuring-your-tensorflow-models/
# 
"""



def read_file_time_sequences(fname):
    sequences = []
    # ignore id number in the beginning of each line
    split_line = lambda l: [float(x) for x in l.split()][1:]
    with open(fname) as f:
        dtype = 0
        for l in f:
            if dtype == 0:
                sequence_tuple = [] #x_in, t_in, y_out, t_out
                sequence_tuple.append(split_line(l))
                dtype += 1
            elif dtype == 1:
                sequence_tuple.append(split_line(l))
                dtype += 1
            elif dtype == 2:
                sequence_tuple.append(split_line(l))
                dtype += 1
            elif dtype == 3:
                sequence_tuple.append(split_line(l))
                sequences.append(sequence_tuple)
                dtype = (dtype + 1) % 4

    return np.array(sequences)



def load_data(dir, sort_by_len=True, valid_ratio=0.1):
    """
    Reads the directory for test and train datasets.
    Divides test set into validation set and test set.
    Sorts all the datasets by their length to make padding
    a little more efficient.
    
    Returns: train, valid, test
    """
    train_set = read_file_time_sequences(dir + '.train')
    test_set = read_file_time_sequences(dir + '.test')
    
    # make validation set from test set before sorting by length
    valid_n = int(len(train_set)*valid_ratio)
    random.shuffle(train_set)
    valid_set = train_set[0:valid_n]
    train_set = train_set[valid_n:]
    
    # sort each set by length to minimize padding in the future
    if sort_by_len:
        sorted_indeces = lambda seq: sorted(range(len(seq)), key=lambda x: len(seq[x][0]))
        reorder = lambda seq, order: [seq[i] for i in order]
        train_set = reorder(train_set, sorted_indeces(train_set))
        test_set = reorder(test_set, sorted_indeces(test_set))
        valid_set = reorder(valid_set, sorted_indeces(valid_set))

    #print len(train_set), len(test_set), len(valid_set)
    return train_set, valid_set, test_set


def get_minibatches_ids(n, minibatch_size, shuffle=True):
    """
    Shuffle dataset at each iteration and get minibatches
    
    Returns: [[1,2,3...], ...] - set of minibatch ids 
    """
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return minibatches


def prepare_data(ox, oxt, oy, oyt, maxlen=None, extended_len=0):
    """
    Pads each sequences with zeroes until maxlen to make a
    minibatch matrix that's of dimension (maxlen, batch_size)
    We use the mask later to mask the loss function
    Returns: padded data & mask that tells us which are fake
            (n_steps, batch_size) for everything
    """
    lengths = [len(seq) for seq in ox]
    # discard if too long
    if maxlen > 0:
        new_lengths = []
        new_ox = []
        new_oxt = []
        new_oy = []
        new_oyt = []
        for l, lox, loxt, loy, loyt in zip(lengths, ox, oxt, oy, oyt):
            if l < maxlen:
                new_lengths.append(l)
                new_ox.append(lox)
                new_oxt.append(loxt)
                new_oy.append(loy)
                new_oyt.append(loyt)
        lengths = new_lengths
        ox = new_ox
        oxt = new_oxt
        oy = new_oy
        oyt = new_oyt
    
    
    maxlen = np.max(lengths)

    # extend to maximal length, TODO: remove
    if extended_len != 0:
        maxlen = extended_len

    batch_size = len(ox)
    
    x = np.zeros((batch_size, maxlen)).astype('int64')
    xt = np.zeros((batch_size, maxlen)).astype(np.float32)
    y = np.zeros((batch_size, maxlen)).astype('int64')
    yt = np.zeros((batch_size, maxlen)).astype(np.float32)
    x_mask = np.zeros((batch_size, maxlen)).astype(np.float32)
    
    for i in range(len(ox)):
        x[i, :lengths[i]] = ox[i]
        xt[i, :lengths[i]] = oxt[i]
        y[i, :lengths[i]] = oy[i]
        yt[i, :lengths[i]] = oyt[i]
        x_mask[i, :lengths[i]] = 1.0
         
    return x, xt, y, yt, x_mask, maxlen


def embed_one_hot(batch_array, depth, length):
    """
    Input: batch_y of shape (batch_size, n_steps)
    Output: batch_y 1-hot-embedded of shape(batch_size, n_steps, n_classes)
    """
    batch_array = np.array(batch_array)
    batch_size, _ = batch_array.shape
    
    one_hot_matrix = np.zeros((batch_size, length, depth))
    for i,array in enumerate(batch_array):
        array = [x - 1 for x in array]
        one_hot_matrix[i, np.arange(len(array)), array] = 1
    return one_hot_matrix

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

    

