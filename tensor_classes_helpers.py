import tensorflow as tf
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
3) output = tf.py_func(func_of_interest)
    the output of py_func needs to be returned in order for func_of_interest to ever be "executed"
4) tf.Print() - not sure how to use. It seems like it still needs to be evaluated with the session.
5) how to get shape of dynamic dimension? - can't.

Tasks:
 - figure out NaN's issue
 - use scopes: https://github.com/llSourcell/tensorflow_demo/blob/master/board.py
*Lab number: 1b11
2) make "ensemble", individual examples
3) new dataset (10% prev=curr, LSTM
4) symmetric (+ & -) activations (to make learn faster) tanh > sigm
5) embedding attempt
"""

def get_tensor_by_name(name):
    print tf.global_variables()
    return [v for v in tf.global_variables() if v.name == name][0]

def input_placeholder(max_length_seq=100, 
                        frame_size=3, name=None):
    
    x = tf.placeholder("float", 
                        [None, max_length_seq, 
                        frame_size], name=name) #None - for dynamic batch sizing
    return x

def output_placeholder(max_length_seq=100, 
                        number_of_classes=50, name=None):
    
    y = tf.placeholder("float", 
                        [None, max_length_seq,  
                        number_of_classes], name=name)
    return y

def weights_init(n_input, n_output, name=None, positive=False):
    init_matrix = None
    if positive:
        init_matrix = tf.random_uniform([n_input, n_output], minval=0.1, maxval=0.99, dtype=tf.float32)
    else:
        init_matrix = tf.random_normal([n_input, n_output])

    W = tf.Variable(init_matrix, name=name)
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
    # x_set: [max_length, batch_size, frpoame_size]
    # splits accross 2nd axis, into 3 splits of x_set tensor (very backwards argument arrangement)
    x, xt, yt = tf.split(x_set, 3, 2)

    # at this point x,xt,yt : [max_length, batch_size, 1] => collapse
    x = tf.reduce_sum(x, reduction_indices=2)
    #xt = tf.reduce_sum(xt, reduction_indices=2)
    #yt = tf.reduce_sum(yt, reduction_indices=2)

    # one hot embedding of x (previous state)
    x = tf.cast(x, tf.int32) # needs integers for one hot embedding to work
    # depth=n_classes, by default 1 for active, 0 inactive, appended as last dimension
    x_vectorized = tf.one_hot(x, ops['n_classes'], name='x_vectorized')
    # x_vectorized: [max_length, batch_size, n_classes]
    return x_vectorized, xt, yt

def errors_and_losses(sess, P_x, P_y, P_len, P_mask, P_batch_size, T_accuracy,  T_cost, dataset_names, datasets, ops):
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
                                                        P_y: DH.embed_one_hot(batch_y, 0.0, ops['n_classes'], ops['max_length']),
                                                        P_len: batch_maxlen,
                                                        P_mask: mask,
                                                        P_batch_size: batch_size})
            acc_avg += accuracy_batch
            loss_avg += cost_batch
        accuracy_entry.append(acc_avg/len(batch_indeces_arr))
        losses_entry.append(cost_batch/len(batch_indeces_arr))
    return accuracy_entry, losses_entry
    

def RNN(x_set, T_W, T_b, T_seq_length, n_hidden, ops):
    # lstm cell
    #lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # get lstm_cell's output
    # dynamic_rnn return by default: 
    #   outputs: [max_time, batch_size, cell.output_size]
    x, xt, yt = cut_up_x(x_set, ops)

    x = tf.concat([x, xt, yt], 2) #[max_time, batch_size, n_hid + 2]
    outputs, states = tf.nn.dynamic_rnn(
                                lstm_cell, 
                                x, 
                                dtype=tf.float32,
                                sequence_length=T_seq_length,
                                time_major=True)
    
    # linear activation, using rnn innter loop last output
    # project into class space: x-[max_time, hidden_units], T_W-[hidden_units, n_classes]
    output_projection = lambda x: tf.nn.softmax(tf.matmul(x, T_W) + T_b)

    return tf.map_fn(output_projection, outputs)


def HPM_params_init(ops):
    # W_in: range of each element is from 0 to 1, since each weight is a "probability" for each hidden unit.
    # W_recurrent:
    W = {'in': weights_init(n_input=ops['n_classes'],
                            n_output=ops['n_hidden'],
                            name='W_in',
                            positive=True),
         'recurrent': weights_init(n_input=ops['n_hidden'],
                                   n_output=ops['n_hidden'],
                                   name='W_recurrent',
                                   positive=True)}

    timescales = 2.0 ** np.arange(1,3)#(-7,7) vs (0, 1)
    n_timescales = len(timescales)

    # tensorflow automacally extends dimension of vector to the rest of dimensions
    # as long the last dimensions agree [x,x,a] + [a] = [x+a, x+a, a+a]
    # TODO: mu - small (1e-3), no softmax for mu, alpha
    mu = tf.random_uniform([n_timescales], minval=1e-4, maxval=1e-3, dtype=tf.float32, name='mu')
    gamma = 1.0 / timescales
    alpha = tf.random_uniform([n_timescales], minval=1e-1, maxval=1.0, dtype=tf.float32, name='alpha')

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
def HPM(x_set, ops, params, batch_size, T_sess):
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
    batch_size = tf.cast(batch_size, tf.int32) #cast placeholder into integer



    def _debugging_function(vals):
        #print "Inside debugging f-n:"
        #print vals
        return False

    def _C(prior_of_event, likelyhood):
        # formula 3 - reweight the ensemble
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
        # Probability of no event occuring at h_prev intensity till delta_t
        # time passes
        # formula 1
        h_prev_tr = tf.transpose(h_prev, [1,0,2])
        result = tf.exp((-(h_prev_tr - mu)*(1.0 - tf.exp(-gamma * delta_t)))/gamma -  mu*delta_t)
        # rotate back [n_hid, batch, n_time] -> [batch, n_hid, n_time]
        return tf.transpose(result, [1,0,2], name='Z')

    def _H(h_prev, delta_t):
        # decay current intensity

        h_prev_tr = tf.transpose(h_prev, [1,0,2]) #[bath_size, n_hid, n_timescales] -> [n_hid, batch_size, n_timescales}
        # gamma * delta_t: [batch_size, n_timescales]
        result = mu + tf.exp(-gamma * delta_t) * (h_prev_tr - mu)
        return tf.transpose(result, [1,0,2], name='H')

    def _y_hat(z, c):
        # (batch_size, n_hidden, n_timescales)
        # output: (batch_size, n_hidden)
        # c - timescale probability
        # z - quantity
        # TODO: remap to some function in valid range
        return tf.reduce_sum(z * c, reduction_indices = [2], name='yhat')

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
        # TODO: activation function choice experiment
        # TODO: add bias for _in (initial probability of which event happened bias)
        event = tf.sigmoid(
                        tf.matmul(x, W['in']))  #:[batch_size, n_classes]*[n_classes, n_hid]

        # recurrent part: since y_hat is for t+1, we wait until here to calculate it rather
        #                   rather than in previous iteration
        y_hat = _y_hat(z, c_) # :(batch_size, n_hidden)
        # TODO: why no bias I forgot?
        event += tf.sigmoid(
                        tf.matmul(y_hat, W['recurrent']))  #:(batch_size, n_hid)*(n_hid, n_hid)

        # 2) update c
        event = tf.expand_dims(event, 2) # make [batch_size, n_hid] into [batch_size, n_hid, 1]
        # to support multiplication by [batch_size, n_hid, n_timescales]
        # TODO: check Mike's code on c's update
        c = event * _C(z*h, c_) + (1.0 - event) * _C(z, c_) # h^0 = 1

        # 3) update intensity
        h += alpha * event

        # 4) apply mask & predict next event
        z_hat = _Z(h, yt)
        h_hat = _H(h, yt)

        y_predict = _y_hat(z_hat, c)
        return [h, c, y_predict]


    x, xt, yt = cut_up_x(x_set, ops)
    # batch_size = tf.shape(x)[1]
    # x: [max_length, batch_size, n_classes]
    # xt, yt: [max_length, batch_size, 1]
    # c0 = np.full([batch_size, ops['n_hidden'], n_timescales], 0.1, np.float32)
    rval = tf.scan(_step,
                    elems=[x, xt, yt],
                    initializer=[
                        tf.zeros([batch_size, ops['n_hidden'], n_timescales], tf.float32) + mu*2, #h
                        tf.fill([batch_size, ops['n_hidden'], n_timescales], 1.0/n_timescales), #c
                        tf.zeros([batch_size, ops['n_hidden']], tf.float32) # yhat

                    ]
                   )

    #debug_print_op = tf.py_func(_debugging_function, [[mu, alpha]], [tf.bool])
    #return predictions
    #rval[2]: [max_length, batch_size, n_hid] - at this point we have
    # all the intensities [n_hid] telling us which event is most likely to fire.

    #tf.summary.histogram('y_hat', rval[2])
    return rval[2]
    # return [rval[2], debug_print_op]
