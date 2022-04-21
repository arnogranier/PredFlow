import os ; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf 
import matplotlib.pyplot as plt 

def drelu(x):
    return tf.cast(tf.greater_equal(x, tf.constant(0.)), tf.float32)

def reduce_batch_outer(x, y):
    return tf.reduce_sum(tf.einsum('nx,ny->nxy',tf.squeeze(x),tf.squeeze(y)), 0)

def one_hot_pred_accuracy(p, t, axis=1):
    binary_comp = tf.argmax(p, axis=axis) == tf.argmax(t, axis=axis)
    count = tf.math.count_nonzero(binary_comp)
    return tf.cast(count, tf.int32) / tf.shape(p)[0]

def mse(v, t):
    return tf.reduce_mean(tf.square(v-t))

@tf.function
def learn(w, p, data, target, lr=0.0005, pr=0.001, f=tf.nn.relu, only_diag=False):
    e = tf.subtract(data, tf.matmul(w, f(target)))
    if only_diag:
        epsilon = tf.linalg.diag(tf.linalg.diag_part(p)) @ e
    else:
        epsilon = tf.matmul(p, e)
    w.assign_add(tf.scalar_mul(lr, reduce_batch_outer(epsilon, f(target))))
    p.assign_add(tf.scalar_mul(pr, tf.eye(tf.shape(p)[0]) - reduce_batch_outer(epsilon, e)/batch_size))

@tf.function
def infer(w, p, data, ir=0.02, T=40, f=tf.nn.relu, df=drelu, only_diag=False):
    r = 0.1*tf.ones((10000, 10, 1))
    
    for _ in range(T):
        e = tf.subtract(data, tf.matmul(w, f(r)))
        if only_diag:
            epsilon = tf.linalg.diag(tf.linalg.diag_part(p)) @ e
        else:
            epsilon = tf.matmul(p, e)
        r += tf.scalar_mul(ir, tf.matmul(w, epsilon, transpose_a=True) * df(r))
            
    return r

if __name__ == "__main__":
    
    # Hyperparamaters
    batch_size = 50
    mlp_architecture = [784, 10]
    n_epochs = 10
    w_init_std = 0.001
    
    # Load MNIST dataset
    import tensorflow_datasets as tfds
    def preprocess(image, label): 
        return (tf.reshape(tf.cast(image, tf.float32), [784, 1]) / 255.,
                tf.cast(tf.expand_dims(tf.one_hot(label, 10), -1), tf.float32))
    ds = tfds.load('mnist', split='train', as_supervised=True)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    # Load test dataset
    test_ds = tfds.load('mnist', split='test', as_supervised=True)
    test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().batch(10000).prefetch(tf.data.experimental.AUTOTUNE)
    test_images, test_targets = test_ds.get_single_element()    
    
    # Training
    full_acc = []
    diag_acc = []
    no_acc = []
    
    # Parameters
    w = tf.Variable(tf.random.normal([784, 10], stddev=w_init_std))
    p = tf.Variable(tf.eye(784))
    
    # Init 
    l = infer(w, p, test_images)
    full_acc.append(one_hot_pred_accuracy(l, test_targets))
    
    # Training
    for epoch in range(n_epochs):
        ds.shuffle(60000)
        for image, target in ds:
            learn(w, p, image+tf.random.normal(tf.shape(image),0,1), target, pr=0.0002, only_diag=False)
        l = infer(w, p, test_images, only_diag=False)
        full_acc.append(one_hot_pred_accuracy(l, test_targets))
    
    # Parameters
    w = tf.Variable(tf.random.normal([784, 10], stddev=w_init_std))
    p = tf.Variable(tf.eye(784))
    
    # Init 
    l = infer(w, p, test_images)
    diag_acc.append(one_hot_pred_accuracy(l, test_targets))

    # Training
    for epoch in range(n_epochs):
        ds.shuffle(60000)
        for image, target in ds:
            learn(w, p, image+tf.random.normal(tf.shape(image),0,1), target, pr=0.0002, only_diag=True)
        l = infer(w, p, test_images, only_diag=True)
        diag_acc.append(one_hot_pred_accuracy(l, test_targets))
        
    # Parameters
    w = tf.Variable(tf.random.normal([784, 10], stddev=w_init_std))
    p = tf.Variable(tf.eye(784))
    
    # Init 
    l = infer(w, p, test_images)
    no_acc.append(one_hot_pred_accuracy(l, test_targets))

    # Training
    for epoch in range(n_epochs):
        ds.shuffle(60000)
        for image, target in ds:
            learn(w, p, image+tf.random.normal(tf.shape(image),0,1), target, pr=0)
        l = infer(w, p, test_images)
        no_acc.append(one_hot_pred_accuracy(l, test_targets))
    
    # Plotting
    plt.plot(full_acc, 'red', lw=2, label='full precision')
    plt.plot(diag_acc, 'orange', lw=2, label='diagional precision')
    plt.plot(no_acc, 'gold', lw=2, label='no precision')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('test accuracy')
    plt.xticks(range(11))
    ax = plt.gca()
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    plt.xlim(0,10)
    plt.ylim(0,1)
    plt.show()