import os ; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf 
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def drelu(x):
    return tf.cast(tf.greater_equal(x, tf.constant(0.)), tf.float32)

def reduce_batch_outer(x, y):
    return tf.reduce_sum(tf.einsum('nx,ny->nxy',tf.squeeze(x),tf.squeeze(y)), 0)

@tf.function
def learn(w, p, data, target, lr=0.02, pr=0.001, f=tf.nn.relu):
    b = tf.cast(tf.shape(data)[0], tf.float32)
    e = tf.subtract(data, tf.matmul(w, f(target)))
    epsilon = tf.matmul(p, e)
    
    w.assign_add(tf.scalar_mul(lr, reduce_batch_outer(epsilon, f(target))))
    p.assign_add(tf.scalar_mul(pr, tf.eye(tf.shape(p)[0]) - reduce_batch_outer(epsilon, e)/b))

@tf.function
def infer(w, p, data, target_size, ir=0.05, T=60, f=tf.nn.relu, df=drelu):
    r = 0.1 * tf.ones((tf.shape(data)[0], target_size, 1))
    
    for _ in range(T):
        e = tf.subtract(data, tf.matmul(w, f(r)))
        epsilon = tf.matmul(p, e)
        r += tf.scalar_mul(ir, tf.matmul(w, epsilon, transpose_a=True) * df(r))
    return r

if __name__ == "__main__":
    
    # Hyperparamaters
    batch_size = 50
    n_epochs = 1
    w_init_std = 0.001
    
    # Load MNIST dataset
    def preprocess(image, label): 
        return (tf.reshape(tf.cast(image, tf.float32), [28*28, 1]) / 255.,
                tf.cast(tf.expand_dims(tf.one_hot(label, 10), -1), tf.float32))
    ds = tfds.load('mnist', split='train', as_supervised=True)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    # Shallow network parameters
    w = tf.Variable(tf.random.normal([28*28, 10], stddev=w_init_std))
    p = tf.Variable(tf.eye(28*28))
    
    # Training
    for epoch in range(n_epochs):
        ds.shuffle(60000)
        for image, target in ds:
            learn(w, p, image + tf.random.normal(tf.shape(image),0,1),
                  target, pr=0.0001)
    
    # Setting the diagonal of the precision to 0 to more easily visualize the extradiagonal terms
    p.assign_add(-tf.linalg.diag(tf.linalg.diag_part(p)))
    
    # Plotting
    fig, axs = plt.subplots(2,2)
    bounds = tf.linspace(-0.05, 0.05, 12)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    cmap = 'RdBu_r'
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    
    def plot_m(ax, k):
        ax.imshow(tf.reshape(p[k,:], (28, 28)), cmap='RdBu_r', norm=norm)
        ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
    plot_m(axs[0,0], 28 * 20 + 18)
    plot_m(axs[0,1], 28 * 14 + 14)
    plot_m(axs[1,0], 28 * 8 + 16)
    plot_m(axs[1,1], 28 * 8 + 10)
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(sm, cax=cbar_ax, ticks=[-0.05, 0, 0.05])
    cbar_ax.text(-0.3, 0.5, 'weight', horizontalalignment='center',
                 verticalalignment='center', rotation='vertical',
                 transform=cbar_ax.transAxes)
    
    plt.show()
    
