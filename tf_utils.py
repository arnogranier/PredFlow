import tensorflow as tf

class Dense(tf.Module):
    def __init__(self, input_dim, output_size, name=None, activation=tf.nn.relu):
        super(Dense, self).__init__(name=name)
        self.w = tf.Variable(tf.random.normal([output_size, input_dim], stddev=.001), name='w')
        #self.b = tf.Variable(tf.zeros([output_size,1]), name='b')
        self.activation = activation
    def __call__(self, x):
        return tf.matmul(self.w, self.activation(x)) #+ self.b

def load_tensorboard_graph(logdir, func, args, name, step=0, kwargs={}):
    writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on()
    func(*args, **kwargs)
    with writer.as_default():
        tf.summary.trace_export(
            name=name,
            step=step,
            profiler_outdir=logdir)

def reduced_batched_outer_product(x, y):
    with tf.name_scope("ReducedBatchedOuterProduct") as scope:
        return tf.reduce_sum(tf.einsum('nx,ny->nxy', tf.squeeze(x), tf.squeeze(y)), 0)

def mlp_weights(*args):
    return [Dense(s1, s2) for (s1, s2) in zip(list(args)[:-2], list(args)[1:-1])] + [Dense(list(args)[-2], list(args)[-1], activation=tf.nn.relu),]

def one_hot_pred_accuracy(p, t):
    return tf.cast(tf.math.count_nonzero(tf.argmax(p, axis=1) == tf.argmax(t, axis=1)), tf.int32)/tf.shape(p)[0]