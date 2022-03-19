import os ; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf 
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

def drelu(x):
    return tf.cast(tf.greater_equal(x, tf.constant(0.)), tf.float32)

def reduce_batch_outer(x, y):
    return tf.reduce_sum(tf.einsum('nx,ny->nxy',tf.squeeze(x),tf.squeeze(y)), 0)

@tf.function
def learn(w, data, target, ir=0.05, lr=0.005, T=20, f=tf.nn.relu, df=drelu):
    N = len(w)
    
    # Initialization
    r = [target,]
    for i in reversed(range(1, N)):
        r.insert(0, tf.matmul(w[i], f(r[0])))
    r.insert(0, data)
    e = [tf.zeros(tf.shape(r[i])) for i in range(N)]
    
    # Inference
    for _ in range(T):
        for i in range(N):
            e[i] = tf.subtract(r[i], tf.matmul(w[i], f(r[i+1])))
        for i in range(1, N): 
            r[i] += tf.scalar_mul(ir, -e[i] + tf.matmul(w[i-1], e[i-1], transpose_a=True) * df(r[i]))
    
    # Learning
    for i in range(N):
        e[i] = tf.subtract(r[i], tf.matmul(w[i], f(r[i+1])))
        w[i].assign_add(tf.scalar_mul(lr, reduce_batch_outer(e[i], f(r[i+1]))))

@tf.function
def generate(w, target, ir=0.05, T=40, f=tf.nn.relu, df=drelu):
    N = len(w)
    
    # Initialization
    r = [target,]
    for i in reversed(range(N)):
        r.insert(0, tf.matmul(w[i], f(r[0])))
    e = [tf.zeros(tf.shape(r[i])) for i in range(N)]
    
    # Inference
    for _ in range(T):
        for i in range(N):
            e[i] = tf.subtract(r[i], tf.matmul(w[i], f(r[i+1])))
        for i in range(1, N): 
            r[i] += tf.scalar_mul(ir, -e[i] + tf.matmul(w[i-1], e[i-1], transpose_a=True) * df(r[i]))
        r[0] += tf.scalar_mul(ir, -e[0])
    
    return r

if __name__ == "__main__":
    
    # Hyperparamaters
    batch_size = 50
    mlp_architecture = [784, 256, 64, 10]
    n_epochs = 5
    w_init_std = 0.001

    # Load MNIST dataset
    def preprocess(image, label): 
        return (tf.reshape(tf.cast(image, tf.float32), [784, 1]) / 255.,
                tf.cast(tf.expand_dims(tf.one_hot(label, 10), -1), tf.float32))
    ds = tfds.load('mnist', split='train', as_supervised=True)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # Reversed MLP weights
    w = [tf.Variable(tf.random.normal([s1, s2], stddev=w_init_std))
         for (s1, s2) in zip(mlp_architecture[:-1], mlp_architecture[1:])]

    # Training
    for epoch in range(n_epochs):
        ds.shuffle(60000)
        for image, target in ds:
            learn(w, image, target)
    
    # Generation
    targets = tf.constant(tf.expand_dims(tf.eye(10), -1))
    l = generate(w, targets)

    # Plotting
    fig, _ = plt.subplots(2,5)
    for i, ax in enumerate(fig.axes):
        ax.imshow(tf.reshape(l[0][i,:,:], (28,28)), cmap="Greys")
        ax.axis("off")
    plt.tight_layout()
    plt.show()