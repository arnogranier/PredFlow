'''
[description]
'''


import tensorflow as tf

class Dense(tf.Module):
    """[summary]

    :param input_dim: dimension of the layer's input
    :type input_dim: int
    :param output_size: dimension of the layer's output
    :type output_size: int
    :param name: custom name for the layer, defaults to None
    :type name: str, optional
    :param activation: activation function, defaults to tf.nn.relu
    :type activation: function, optional
    :param stddev: standard deviation of the normal initialization, defaults to .001
    :type stddev: float, optional
    """

    def __init__(self, input_dim, output_size, name=None, activation=tf.nn.relu, stddev=.001):
        super(Dense, self).__init__(name=name)
        self.w = tf.Variable(tf.random.normal([output_size, input_dim], stddev=stddev), name='w')
        self.activation = activation
        
    def __call__(self, x):
        """[summary]

        :param x: input
        :type x: 3d tf.Tensor of float32
        :return: output
        :rtype: 3d tf.Tensor of float32
        """
        
        return tf.matmul(self.w, self.activation(x)) 
    
class BiasedDense(tf.Module):
    """[summary]

    :param input_dim: dimension of the layer's input
    :type input_dim: int
    :param output_size: dimension of the layer's output
    :type output_size: int
    :param name: custom name for the layer, defaults to None
    :type name: str, optional
    :param activation: activation function, defaults to tf.nn.relu
    :type activation: function, optional
    :param stddev: standard deviation of the normal initialization, defaults to .001
    :type stddev: float, optional
    """
        
    def __init__(self, input_dim, output_size, name=None, activation=tf.nn.relu, stddev=.001):
        super(BiasedDense, self).__init__(name=name)
        self.w = tf.Variable(tf.random.normal([output_size, input_dim], stddev=stddev), name='w')
        self.b = tf.Variable(tf.zeros([output_size,1]), name='b')
        self.activation = activation
        
    def __call__(self, x):
        """[summary]

        :param x: input
        :type x: 3d tf.Tensor of float32
        :return: output
        :rtype: 3d tf.Tensor of float32
        """
        
        return tf.matmul(self.w, self.activation(x)) + self.b

def load_tensorboard_graph(logdir, func, args, name, step=0, kwargs={}):
    """[summary]

    :param logdir: log folder path
    :type logdir: str
    :param func: function to analyze
    :type func: function
    :param args: arguments of func
    :type args: list
    :param name: name of the tensorboard trace
    :type name: str
    :param step: tensorboard step, defaults to 0
    :type step: int, optional
    :param kwargs: kwargs of func, defaults to {}
    :type kwargs: dict, optional
    """
    
    writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on()
    func(*args, **kwargs)
    with writer.as_default():
        tf.summary.trace_export(
            name=name,
            step=step,
            profiler_outdir=logdir)

def reduced_batched_outer_product(x, y):
    """[summary]

    :param x: first tensor
    :type x: 3d tf.Tensor
    :param y: second tensor
    :type y: 3d tf.Tensor
    :return: outer product summed over batch dimesion 
    :rtype: 2d tf.Tensor
    """
    
    with tf.name_scope("ReducedBatchedOuterProduct"):
        return tf.reduce_sum(tf.einsum('nx,ny->nxy', tf.squeeze(x), tf.squeeze(y)), 0)

def relu_derivate(x):
    """[summary]

    :param x: input
    :type x: tf.Tensor
    :return: output
    :rtype: tf.Tensor
    """
    
    with tf.name_scope("ReLUDerivate"):
        return tf.cast(tf.greater(x, tf.constant(0.)), tf.float32)

def mlp(*args, biased=False, reversed_flow=False, activation=tf.nn.relu, stddev=0.01, only_return_weights=False):
    """[summary]

    :param args: sequence of int representing layer sizes
    :param biased: controls weither we use bias in layers, defaults to False
    :type biased: bool, optional
    :param reversed_flow: controls weither we reverse the flow of activation (default is bottom-up), defaults to False
    :type reversed_flow: bool, optional
    :param activation: activation function, defaults to tf.nn.relu
    :type activation: function, optional
    :param stddev: standard deviation of the normal initialization, defaults to 0.01
    :type stddev: float, optional
    :param only_return_weights: controls weither we return a list of tf.Module or 2d variable weight matrices, defaults to False
    :type only_return_weights: bool, optional
    :return: model
    :rtype: list of :py:class:`tf_utils.Dense` or :py:class:`tf_utils.BiasedDense` or 2d variable tf.Tensor of float32
    """
    
    if only_return_weights:
        if not reversed_flow:
            return [tf.Variable(tf.random.normal([s2, s1], stddev=stddev)) for (s1, s2) in zip(list(args)[:-1], list(args)[1:])]
        else:
            return [tf.Variable(tf.random.normal([s1, s2], stddev=stddev)) for (s1, s2) in zip(list(args)[:-1], list(args)[1:])]
    else:
        if not biased:
            if not reversed_flow:
                return [Dense(s1, s2, activation=activation, stddev=stddev) for (s1, s2) in zip(list(args)[:-1], list(args)[1:])]
            else:
                return [Dense(s2, s1, activation=activation, stddev=stddev) for (s1, s2) in zip(list(args)[:-1], list(args)[1:])]
        else:
            if not reversed_flow:
                return [BiasedDense(s1, s2, activation=activation, stddev=stddev) for (s1, s2) in zip(list(args)[:-1], list(args)[1:])]
            else:
                return [BiasedDense(s2, s1, activation=activation, stddev=stddev) for (s1, s2) in zip(list(args)[:-1], list(args)[1:])]

def one_hot_pred_accuracy(p, t, axis=1):
    """[summary]

    :param p: network prediction
    :type p: 3d tf.Tensor
    :param t: ground truth target
    :type t: 3d tf.Tensor
    :param axis: argmax axis, defaults to 1
    :type axis: int, optional
    :return: accuracy
    :rtype: float32
    """
    
    with tf.name_scope("AccuracyComputation"):
        return tf.cast(tf.math.count_nonzero(tf.argmax(p, axis=axis) == tf.argmax(t, axis=axis)), tf.int32)/tf.shape(p)[0]