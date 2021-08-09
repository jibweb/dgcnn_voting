import functools
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected as fc


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
    """ Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/
          how-could-i-use-batch-normalization-in-tensorflow

    Args:
        inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        is_training:   boolean tf.Varialbe, true indicates training phase
        scope:         string, variable scope
        moments_dims:  a list of ints, indicating dimensions for moments
                       calculation
        bn_decay:      float or float tensor variable, controling moving
                       average weight
    Return:
        normed:        batch-normalized maps
    """
    with tf.variable_scope(scope):
        num_channels = inputs.get_shape()[-1].value
        beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims,
                                              name='moments')
        decay = bn_decay if bn_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        # Operator that maintains moving averages of variables.
        ema_apply_op = tf.cond(is_training,
                               lambda: ema.apply([batch_mean, batch_var]),
                               lambda: tf.no_op())

        # Update moving average and return current batch's avg and var.
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # ema.average returns the Variable holding the average of var.
        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean),
                                     ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inputs, mean, var,
                                           beta, gamma, 1e-3)
    return normed


def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
    """ Batch normalization on FC data.

    Args:
        inputs:      Tensor, 2D BxC input
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average
                     weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, ], bn_decay)


def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 1D convolutional maps.

    Args:
        inputs:      Tensor, 3D BLC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving
                     average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, 1], bn_decay)


def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 2D convolutional maps.

    Args:
        inputs:      Tensor, 4D BHWC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average
                     weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, 1, 2], bn_decay)


def weight_variable(shape, reg_constant):
    # initializer = tf.truncated_normal_initializer(stddev=0.1)
    return tf.get_variable("weights", shape,
                           initializer=tf.contrib.layers.xavier_initializer(),
                           regularizer=tf.contrib.layers.l2_regularizer(
                                reg_constant))


def bias_variable(shape, reg_constant):
    # initial = tf.constant(0.0, shape=shape)
    # return tf.Variable(initial)
    return tf.get_variable("biases",
                           shape,
                           initializer=tf.constant_initializer(0.0),
                           regularizer=tf.contrib.layers.l2_regularizer(
                                reg_constant))


def fc_bn(tens_inp, filters, scope, is_training, reg_constant,
          bn_decay=None, activation=tf.nn.relu):
    with tf.variable_scope(scope):
        dense = fc(tens_inp, filters,
                   scope="dense",
                   activation_fn=None,
                   weights_regularizer=tf.contrib.layers.l2_regularizer(
                        reg_constant),
                   biases_regularizer=tf.contrib.layers.l2_regularizer(
                        reg_constant))
        bn = batch_norm_for_fc(dense,
                               is_training=is_training,
                               bn_decay=bn_decay,
                               scope="bn")
        if activation:
            return activation(bn)
        else:
            return bn


def conv1d(tens_in, out_sz, reg_constant, scope,
           activation=tf.nn.relu, use_bias=True):
    # Tensor size setup
    in_channels = tens_in.get_shape()[-1].value
    filter_shape = [1, in_channels, out_sz]

    # Variable and operations definition
    with tf.variable_scope(scope):
        kernel = weight_variable(filter_shape, reg_constant)
        feats = tf.nn.conv1d(tens_in, kernel, stride=1,
                             padding='VALID')
        if use_bias:
            biases = bias_variable([out_sz], reg_constant)
            feats += biases

        if activation:
            return activation(feats)
        else:
            return feats


def conv1d_bn(tens_in, out_sz, reg_constant, is_training, scope,
              bn_decay=None, activation=tf.nn.relu, use_bias=True):
    with tf.variable_scope(scope):
        feats = conv1d(tens_in, out_sz, reg_constant, "conv",
                       activation=None, use_bias=use_bias)
        feats_bn = batch_norm_for_conv1d(
                    feats,
                    is_training=is_training,
                    bn_decay=bn_decay,
                    scope='bn')
        if activation:
            return activation(feats_bn)
        else:
            return feats_bn


def conv2d(tens_in, out_sz, kernel_sz, reg_constant, scope,
           activation=tf.nn.relu, use_bias=True):
    # Tensor size setup
    in_channels = tens_in.get_shape()[-1].value
    if type(kernel_sz) == list or type(kernel_sz) == tuple:
        assert len(kernel_sz) == 2
        filter_shape = list(kernel_sz) + [in_channels, out_sz]
    else:
        filter_shape = [kernel_sz, kernel_sz, in_channels, out_sz]

    # Variable and operations definition
    with tf.variable_scope(scope):
        kernel = weight_variable(filter_shape, reg_constant)
        feats = tf.nn.conv2d(tens_in, kernel, strides=[1, 1, 1, 1],
                             padding='VALID')
        if use_bias:
            biases = bias_variable([out_sz], reg_constant)
            feats += biases

        if activation:
            return activation(feats)
        else:
            return feats


def conv2d_bn(tens_in, out_sz, kernel_sz, reg_constant, is_training, scope,
              bn_decay=None, activation=tf.nn.relu, use_bias=True):
    with tf.variable_scope(scope):
        feats = conv2d(tens_in, out_sz, kernel_sz, reg_constant, "conv",
                       activation=None, use_bias=use_bias)
        feats_bn = batch_norm_for_conv2d(
                    feats,
                    is_training=is_training,
                    bn_decay=bn_decay,
                    scope='bn')
        if activation:
            return activation(feats_bn)
        else:
            return feats_bn


def conv3d(tens_in, out_sz, kernel_sz, reg_constant, scope,
           activation=tf.nn.relu, use_bias=True):
    # Tensor size setup
    in_channels = tens_in.get_shape()[-1].value
    if type(kernel_sz) == list or type(kernel_sz) == tuple:
        assert len(kernel_sz) == 3
        filter_shape = list(kernel_sz) + [in_channels, out_sz]
    else:
        filter_shape = [kernel_sz, kernel_sz, kernel_sz, in_channels, out_sz]

    # Variable and operations definition
    with tf.variable_scope(scope):
        kernel = weight_variable(filter_shape, reg_constant)
        feats = tf.nn.conv3d(tens_in, kernel, strides=[1, 1, 1, 1, 1],
                             padding='VALID')

        if use_bias:
            biases = bias_variable([out_sz], reg_constant)
            feats += biases

        if activation:
            return activation(feats)
        else:
            return feats
