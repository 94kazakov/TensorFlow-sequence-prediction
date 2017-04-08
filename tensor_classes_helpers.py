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
6) start: tensorboard --logdir=run1:logs/run1/ --port 6006

Tasks:
 - figure out NaN's issue
 - how to access variable by name?  (ex: I want to retrieve a named variable)
 - use scopes: https://github.com/llSourcell/tensorflow_demo/blob/master/board.py
*Lab number: 1b11
2) make "ensemble", individual examples
3) new dataset (10% prev=curr, LSTM
4) symmetric (+ & -) activations (to make learn faster) tanh > sigm
5) embedding attempt
"""

def get_tensor_by_name(name):
    print tf.all_variables()
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

def weights_init(n_input, n_output, name=None, small=True):
    init_matrix = None
    if small:
        init_matrix = tf.random_normal([n_input, n_output], stddev=0.01)
    else:
        init_matrix = tf.random_normal([n_input, n_output])
    W = tf.Variable(init_matrix, name=name)
    return W

def bias_init(n_output, name=None, small=True):
    b = None
    if small: #bias is negative so that initially, bias is pulling tthe sigmoid towards 0, not 1/2.
        b = tf.Variable(tf.random_normal([n_output], mean=-2.0, stddev = 0.01), name=name)
    else:
        b = tf.Variable(tf.random_normal([n_output]), name=name)
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
                            name='W_in'),
         'recurrent': weights_init(n_input=ops['n_hidden'],
                                   n_output=ops['n_hidden'],
                                   name='W_recurrent',
                                   small=True),
         'out':  weights_init(n_input=ops['n_hidden'],
                                   n_output=ops['n_classes'],
                                   name='W_out')
         }

    b = {'in': bias_init(n_output=ops['n_hidden'],
                         name='b_in',
                         small=False),
         'recurrent': bias_init(n_output=ops['n_hidden'],
                         name='b_recurrent',
                         small=True),
         'out': bias_init(n_output=ops['n_classes'],
                         name='b_out',
                         small=False)
        }

    timescales = 2.0 ** np.arange(-7,7)#(-7,7) vs (0, 1)
    n_timescales = len(timescales)

    # tensorflow automacally extends dimension of vector to the rest of dimensions
    # as long the last dimensions agree [x,x,a] + [a] = [x+a, x+a, a+a]

    mu = tf.random_uniform([n_timescales], minval=1e-4, maxval=1e-3, dtype=tf.float32, name='mu')
    gamma = 1.0 / timescales
    alpha = tf.random_uniform([n_timescales], minval=1e-1, maxval=1.0, dtype=tf.float32, name='alpha')
    c = tf.random_normal([n_timescales], mean=1.0/n_timescales, stddev = 0.05)


    params = {
        'W': W,
        'b': b,
        'timescales': timescales,
        'n_timescales': n_timescales,
        'mu': mu,
        'gamma': gamma,
        'alpha': alpha,
        'c': c
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
    b = params['b']
    timescales = params['timescales']
    n_timescales = params['n_timescales']
    mu = params['mu']
    c_init = params['c']
    c_init = c_init/tf.reduce_sum(c_init)# initialize all timescales equally likely
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
        minimum = 1e-30
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
        return tf.reduce_sum(z * c, reduction_indices = [2], name='yhat')

    def _step(accumulated_vars, input_vars):

        h_, c_, _, _ = accumulated_vars
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


        # recurrent part: since y_hat is for t+1, we wait until here to calculate it rather
        #                   rather than in previous iteration
        y_hat = _y_hat(z, c_) # :(batch_size, n_hidden)

        event = tf.sigmoid(
                        tf.matmul(x, W['in']) +  #:[batch_size, n_classes]*[n_classes, n_hid]
                        tf.matmul(y_hat, W['recurrent']) + b['recurrent'])  #:(batch_size, n_hid)*(n_hid, n_hid)

        event = tf.clip_by_value(event, 0, 1) # TODO: how does Mike avoid this?
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
        return [h, c, y_predict, c]



    x, xt, yt = cut_up_x(x_set, ops)

    # collect all the variables of interest
    #TODO: make this an option. this wastes a lot of time during computation
    tf.summary.histogram('W_in', W['in'], ['W'])
    tf.summary.histogram('W_rec', W['recurrent'], ['W'])
    tf.summary.histogram('W_out', W['out'], ['W'])
    tf.summary.histogram('b_rec', b['recurrent'], ['b'])
    tf.summary.histogram('b_out', b['out'], ['b'])
    tf.summary.histogram('c_init', c_init, ['init'])
    tf.summary.histogram('mu_init', mu, ['init'])
    tf.summary.histogram('alpha_init', alpha, ['init'])
    T_summary_weights = tf.summary.merge([
                            tf.summary.merge_all('W'),
                            tf.summary.merge_all('b'),
                            tf.summary.merge_all('init')
                            ])


    rval = tf.scan(_step,
                    elems=[x, xt, yt],
                    initializer=[
                        tf.zeros([batch_size, ops['n_hidden'], n_timescales], tf.float32) + mu, #h
                        tf.zeros([batch_size, ops['n_hidden'], n_timescales], tf.float32) + c_init, #c
                        tf.zeros([batch_size, ops['n_hidden']], tf.float32), # yhat
                        tf.zeros([batch_size, ops['n_hidden'], n_timescales]) #debugging placeholder

                    ]
                   , name='hpm/scan')

    #debug_print_op = tf.py_func(_debugging_function, [[mu, alpha]], [tf.bool])
    #return predictions
    #rval[2]: [max_length, batch_size, n_hid] - at this point we have
    # all the intensities [n_hid] telling us which event is most likely to fire.

    #tf.summary.histogram('y_hat', rval[2])
    hidden_prediction = tf.transpose(rval[2], [1, 0, 2]) # -> [batch_size, n_steps, n_classes]
    output_projection = lambda x: tf.clip_by_value(tf.nn.softmax(tf.matmul(x, W['out']) + b['out']), 1e-8, 1.0)

    # TODO: make return a dict that's then assigned depending on what options are chosen?
    return tf.map_fn(output_projection, hidden_prediction), rval[3], T_summary_weights
