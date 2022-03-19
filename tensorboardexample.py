import os ; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf 
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

def drelu(x):
    with tf.name_scope("ReLUDerivate"):
        return tf.cast(tf.greater_equal(x, tf.constant(0.)), tf.float32)

def reduce_batch_outer(x, y):
    with tf.name_scope("ReducedBatchOuter"):
        return tf.reduce_sum(tf.einsum('nx,ny->nxy',tf.squeeze(x),tf.squeeze(y)), 0)

@tf.function
def learn(w, data, target, ir=tf.constant(0.05), lr=tf.constant(0.005), T=5, f=tf.nn.relu, df=drelu):
    with tf.name_scope("Initialization"):
        N = len(w)
        
        # Initialization
        r = [target,]
        for i in reversed(range(1, N)):
            r.insert(0, tf.matmul(w[i], f(r[0])))
        r.insert(0, data)
        e = [tf.zeros(tf.shape(r[i])) for i in range(N)]
    
    with tf.name_scope("Inference"):
        # Inference
        for _ in range(T):
            with tf.name_scope("PredictionErrorComputation"):
                for i in range(N):
                    e[i] = tf.subtract(r[i], tf.matmul(w[i], f(r[i+1])))
            with tf.name_scope("RepresentationUpdate"):
                for i in range(1, N): 
                    r[i] += tf.scalar_mul(ir, -e[i] + tf.matmul(w[i-1], e[i-1], transpose_a=True) * df(r[i]))
    
    with tf.name_scope("Learning"):
        # Learning
        for i in range(N):
            with tf.name_scope("PredictionErrorComputation"):
                e[i] = tf.subtract(r[i], tf.matmul(w[i], f(r[i+1])))
            with tf.name_scope("WeightUpdate"):
                w[i].assign_add(tf.scalar_mul(lr, reduce_batch_outer(e[i], f(r[i+1]))))

if __name__ == "__main__":
    
    # Hyperparamaters
    batch_size = 50
    mlp_architecture = [784, 256, 64, 10]

    # Load MNIST dataset
    def preprocess(image, label): 
        return (tf.reshape(tf.cast(image, tf.float32), [784, 1]) / 255.,
                tf.cast(tf.expand_dims(tf.one_hot(label, 10), -1), tf.float32))
    ds = tfds.load('mnist', split='train', as_supervised=True)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # Reversed MLP weights
    w = [tf.Variable(tf.random.normal([s1, s2], stddev=0.001, name='w', dtype=tf.float32))
         for (s1, s2) in zip(mlp_architecture[:-1], mlp_architecture[1:])]

    logdir = '/home/arno/tensorboard_trace/'
    writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on()
    
    # Training for one step
    for image, target in ds:
        learn(w, tf.constant(image), tf.constant(target))
        break
    
    with writer.as_default():
        tf.summary.trace_export(
            name='trace',
            step=0,
            profiler_outdir=logdir)