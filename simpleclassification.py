import os ; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf 

def drelu(x):
    return tf.cast(tf.greater_equal(x, tf.constant(0.)), tf.float32)

def reduce_batch_outer(x, y):
    return tf.reduce_sum(tf.einsum('nx,ny->nxy',tf.squeeze(x),tf.squeeze(y)), 0)

def one_hot_pred_accuracy(p, t, axis=1):
    binary_comp = tf.argmax(p, axis=axis) == tf.argmax(t, axis=axis)
    count = tf.math.count_nonzero(binary_comp)
    return tf.cast(count, tf.int32) / tf.shape(p)[0]

@tf.function
def learn(w, data, target, ir=0.1, lr=0.02, T=20, f=tf.nn.relu, df=drelu):
    N = len(w)
    
    # Initialization
    r = [data, ]
    for i in range(N-1):
        r.append(tf.matmul(w[i], f(r[-1])))
    r.append(target)
    e = [tf.zeros(tf.shape(r[i])) for i in range(1,N+1)]
    
    # Inference
    for _ in range(T):
        for i in range(N):
            e[i] = tf.subtract(r[i+1], tf.matmul(w[i], f(r[i])))
        for i in range(1, N): 
            r[i] += tf.scalar_mul(ir, -e[i-1] + tf.matmul(w[i], e[i], transpose_a=True) * df(r[i]))
    
    # Learning
    for i in range(N):
        e[i] = tf.subtract(r[i+1], tf.matmul(w[i], f(r[i])))
        w[i].assign_add(tf.scalar_mul(lr, reduce_batch_outer(e[i], f(r[i]))))

@tf.function
def infer(w, data, ir=0.1, T=40, f=tf.nn.relu, df=drelu):
    N = len(w)
    
    # Initialization
    r = [data, ]
    for i in range(N-1):
        r.append(tf.matmul(w[i], f(r[-1])))
    r.append(tf.matmul(w[-1], f(r[-1])))
    e = [tf.zeros(tf.shape(r[i])) for i in range(1,N+1)]
    
    # Inference
    for _ in range(T):
        for i in range(N):
            e[i] = tf.subtract(r[i+1], tf.matmul(w[i], f(r[i])))
        for i in range(1, N): 
            r[i] += tf.scalar_mul(ir, -e[i-1] + tf.matmul(w[i], e[i], transpose_a=True) * df(r[i]))
        r[N] += tf.scalar_mul(ir, -e[N-1])
    
    return r

if __name__ == "__main__":
    
    # Hyperparamaters
    batch_size = 50
    mlp_architecture = [784, 256, 64, 10]
    n_epochs = 5
    w_init_std = 0.001

    # Load MNIST dataset
    import tensorflow_datasets as tfds
    def preprocess(image, label): 
        return (tf.reshape(tf.cast(image, tf.float32), [784, 1]) / 255.,
                tf.cast(tf.expand_dims(tf.one_hot(label, 10), -1), tf.float32))
    ds = tfds.load('mnist', split='train', as_supervised=True)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # MLP weights
    w = [tf.Variable(tf.random.normal([s2, s1], stddev=w_init_std))
         for (s1, s2) in zip(mlp_architecture[:-1], mlp_architecture[1:])]

    # Training
    for epoch in range(n_epochs):
        ds.shuffle(60000)
        for image, target in ds:
            learn(w, image, target)
    
    # Load test dataset
    test_ds = tfds.load('mnist', split='test', as_supervised=True)
    test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().batch(10000).prefetch(tf.data.experimental.AUTOTUNE)
    test_images, test_targets = test_ds.get_single_element()
    
    # Classification
    l = infer(w, test_images)
    tf.print(one_hot_pred_accuracy(l[-1], test_targets))