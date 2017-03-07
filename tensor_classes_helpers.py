import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import data_help as DH
import numpy as np

"""
NOTES:
1) when tensor([a,b,c])*tensor([b,c]) = tensor([a,b,c])
        tensor([a,b])*tensor([b]) = tensor([a,b])
        tensor([a,1])*tensor([b]) = tensor([a,b]) - equivalent
2) dynamics shape vs static:
        tf.shape(my_tensor)[0] - dynamics (as graph computes) ex: batch_size=current_batch_size
        my_tensor.get_shape() - static (graph's 'locked in' value) ex: batch_size=?
"""

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

def softmax_init(shape):
    # softmax initialization of size shape casted to tf.float32
    return tf.cast(tf.nn.softmax(tf.Variable(tf.random_normal(shape))), tf.float32)

def cut_up_x(x_set, ops):
    # x_set: [batch_size, max_length, frame_size]
    x_set = tf.transpose(x_set, [1,0,2])
    x_set = tf.cast(x_set, tf.float32)
    # x_set: [max_length, batch_size, frame_size]
    # splits accross 2nd axis, into 3 splits of x_set tensor (very backwards argument arrangement)
    x, xt, yt = tf.split(2, 3, x_set)

    # at this point x,xt,yt : [max_length, batch_size, 1] => collapse
    x = tf.reduce_sum(x, reduction_indices=2)
    #xt = tf.reduce_sum(xt, reduction_indices=2)
    #yt = tf.reduce_sum(yt, reduction_indices=2)

    # one hot embedding of x (previous state)
    x = tf.cast(x, tf.int32) # needs integers for one hot embedding to work
    # depth=n_classes, by default 1 for active, 0 inactive, appended as last dimension
    x_vectorized = tf.one_hot(x, ops['n_classes'])
    # x_vectorized: [max_length, batch_size, n_classes]
    return x_vectorized, xt, yt

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
            x_set, batch_y, batch_maxlen, batch_size, mask = DH.pick_batch(
                                            dataset = dataset,
                                            batch_indeces = batch_ids, 
                                            max_length = ops['max_length']) 
            accuracy_batch, cost_batch = sess.run([T_accuracy, T_cost],
                                                    feed_dict={
                                                        P_x: x_set,
                                                        P_y: DH.embed_one_hot(batch_y, ops['batch_size'], ops['n_classes'], ops['max_length']),
                                                        P_len: batch_maxlen,
                                                        P_mask: mask,
                                                        P_batch_size: batch_size})
            acc_avg += accuracy_batch
            loss_avg += cost_batch
        accuracy_entry.append(acc_avg/len(batch_indeces_arr))
        losses_entry.append(cost_batch/len(batch_indeces_arr))
    return accuracy_entry, losses_entry
    

def RNN(x, T_W, T_b, T_seq_length, n_hidden):
    # TODO change input for LSTM as concatnation vectorx, xt, yt
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


def HPM_params_init(ops):
    W = {'in': weights_init(n_input=ops['n_classes'],
                            n_output=ops['n_hidden']),
         'recurrent': weights_init(n_input=ops['n_hidden'],
                                   n_output=ops['n_hidden'])}

    timescales = 2.0 ** np.arange(-7, 7)
    n_timescales = len(timescales)

    # tensorflow automacally extends dimension of vector to the rest of dimensions
    # as long the last dimensions agree [x,x,a] + [a] = [x+a, x+a, a+a]
    mu = softmax_init([n_timescales])
    gamma = 1.0 / timescales
    alpha = softmax_init([n_timescales])
    params = {
        'W': W,
        'timescales': timescales,
        'n_timescales': n_timescales,
        'mu': mu,
        'gamma': gamma,
        'alpha': alpha
    }
    return params



# HPM logic:
# Learn weights of the hawkes' processes.
# Have multiple timescales for each process that are ready to "kick-in".
# For a certain event type in whichever time-scale works best => reinitialize c_
# every new sequence. 
def HPM(x_set, ops, params, batch_size):
    # init h, alphas, timescales, mu etc
    # convert x from [batch_size, max_length, frame_size] to
    #               [max_length, batch_size, frame_size]
    # and step over each time_step with _step function
    W = params['W']
    timescales = params['timescales']
    n_timescales = params['n_timescales']
    mu = params['mu']
    gamma = params['gamma']
    alpha = params['alpha']


    # x = [batch_x, batch_xt, batch_yt]


    # TODO make x a one hot vector
    def _C(prior_of_event, likelyhood):
        # formula 3
        # likelihood, prior, posterior have dimensions:
        #       [batch_size, n_hid, n_timescales]
        minimum = 1e-5
        # likelyhood = c_
        timescale_posterior = prior_of_event * likelyhood + minimum
        timescale_posterior = timescale_posterior / tf.reduce_sum(timescale_posterior,
                                                                  reduction_indices=[2],
                                                                  keep_dims=True)

        return timescale_posterior

    def _Z(h_prev, delta_t):
        # delta_t: batch_size x n_timescales
        # h_prev: batch_size x n_hid x n_timescales
        # Probability of not event occuring at h_prev intensity till delta_t
        # time passes
        # formula 1
        #TODO: where is mu in Mike's code?
        h_prev_tr = tf.transpose(h_prev, [1,0,2])
        result = tf.exp((-(h_prev_tr - mu)*(1.0 - tf.exp(-gamma * delta_t)))/gamma -  mu*delta_t)
        # rotate back [n_hid, batch, n_time] -> [batch, n_hid, n_time]
        return tf.transpose(result, [1,0,2])

    def _H(h_prev, delta_t):
        # decay current intensity

        h_prev_tr = tf.transpose(h_prev, [1,0,2]) #[bath_size, n_hid, n_timescales] -> [n_hid, batch_size, n_timescales}
        # gamma * delta_t: [batch_size, n_timescales]
        result = mu + tf.exp(-gamma * delta_t) * (h_prev_tr - mu)
        return tf.transpose(result, [1,0,2])

    def _y_hat(z, c):
        # (batch_size, n_hidden, n_timescales)
        # output: (batch_size, n_hidden)
        # TODO: remap to some function in valid range
        return tf.reduce_sum(z * c, reduction_indices = [2])

    def _step(accumulated_vars, input_vars):

        h_, c_, _ = accumulated_vars
        x, xt, yt = input_vars
        # : mask: (batch_size, n_classes
        # : x - vectorized x: (batch_size, n_classes)
        # : xt, yt: (batch_size, 1)
        # : h_, c_ - from previous iteration: (batch_size, n_hidden, n_timescales)

        # 1) event
        # current z, h
        h = _H(h_, xt)
        z = _Z(h_, xt) #(batch_size, n_hidden, n_timescales)
        # input part:
        event = tf.matmul(x, W['in'])  #:[batch_size, n_classes]*[n_classes, n_hid]

        # recurrent part: since y_hat is for t+1, we wait until here to calculate it rather
        #                   rather than in previous iteration
        y_hat = _y_hat(z, c_) # :(batch_size, n_hidden)
        # TODO: why no bias I forgot?
        event += tf.matmul(y_hat, W['recurrent'])  #:(batch_size, n_hid)*(n_hid, n_hid)

        # 2) update c
        event = tf.expand_dims(event, 2) # make [batch_size, n_hid] into [batch_size, n_hid, 1]
        # to support multiplication by [batch_size, n_hid, n_timescales]
        c = event * _C(z*(h + mu), c_) + (1.0 - event) * _C(z, c_) # h^0 = 1

        # 3) update intensity
        h += alpha * event

        # 4) apply mask & predict next event
        z_hat = _Z(h, yt)
        h_hat = _H(h, yt)

        y_predict = _y_hat(z_hat, c)
        return [h, c, y_predict]


    x, xt, yt = cut_up_x(x_set, ops)
    #TODO: how to get shape of dynamic dimension?
    # batch_size = tf.shape(x)[1]

    print "Batch", batch_size
    # x: [max_length, batch_size, n_classes]
    # xt, yt: [max_length, batch_size, 1]
    c0 = np.full([batch_size, ops['n_hidden'], n_timescales], 0.1, np.float32)
    rval = tf.scan(_step,
                    elems=[x, xt, yt],
                    initializer=[
                        np.zeros([batch_size, ops['n_hidden'], n_timescales], np.float32),
                        c0, #TODO c_ initliazation according to our formulas
                        np.zeros([batch_size, ops['n_hidden']], np.float32)
                    ] # h, c, yhat
                   )




    #return predictions
    print "predictions:", rval[2] #[max_length, batch_size, n_hid] - at this point we have
    # all the intensities [n_hid] telling us which event is most likely to fire.
    return rval[2]
