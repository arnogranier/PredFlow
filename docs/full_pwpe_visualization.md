---
title: Visualization the precision matrix
tagline: Full precision-weighting of prediction errors in pixel space of a shallow network on MNIST - Visualization the precision matrix
---

Import tensorflow:
```python
import tensorflow as tf 
```

Define the derivate of the ReLU function: 
```python
def drelu(x):
    return tf.cast(tf.greater_equal(x, tf.constant(0.)), tf.float32)
```

Define a function to compute the reduced sum (over batches) of batched outer products: 
```python
def reduce_batch_outer(x, y):
    return tf.reduce_sum(tf.einsum('nx,ny->nxy',tf.squeeze(x),tf.squeeze(y)), 0)
```

Define a step of learning for a supervised generative shallow predictive coding network with weights `w` and precision matrix `p`
```python
@tf.function
def learn(w, p, data, target, lr=0.02, pr=0.001, f=tf.nn.relu):
    b = tf.cast(tf.shape(data)[0], tf.float32)
    e = tf.subtract(data, tf.matmul(w, f(target)))
    epsilon = tf.matmul(p, e)
    
    w.assign_add(tf.scalar_mul(lr, reduce_batch_outer(epsilon, f(target))))
    p.assign_add(tf.scalar_mul(pr, tf.eye(tf.shape(p)[0]) - reduce_batch_outer(epsilon, e)/b))
```

Define an inference loop for a supervised generative shallow predictive coding network with weights `w` and precision matrix `p`
```python
@tf.function
def infer(w, p, data, target_size, ir=0.05, T=60, f=tf.nn.relu, df=drelu):
    r = 0.1 * tf.ones((tf.shape(data)[0], target_size, 1))
    
    for _ in range(T):
        e = tf.subtract(data, tf.matmul(w, f(r)))
        epsilon = tf.matmul(p, e)
        r += tf.scalar_mul(ir, tf.matmul(w, epsilon, transpose_a=True) * df(r))
    return r
```

Hyperparamaters: 
```python
batch_size = 50
n_epochs = 10
w_init_std = 0.001
```

Load the MNIST dataset: 
```python
import tensorflow_datasets as tfds
def preprocess(image, label): 
    return (tf.reshape(tf.cast(image, tf.float32), [28*28, 1]) / 255.,
            tf.cast(tf.expand_dims(tf.one_hot(label, 10), -1), tf.float32))
ds = tfds.load('mnist', split='train', as_supervised=True)
ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.cache().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
```

Shallow network parameters:
```python
w = tf.Variable(tf.random.normal([28*28, 10], stddev=w_init_std))
p = tf.Variable(tf.eye(28*28))
```

Training loop: 
```python
for epoch in range(n_epochs):
    ds.shuffle(60000)
    for image, target in ds:
        learn(w, p, image + tf.random.normal(tf.shape(image),0,1),
                target, pr=0.0001)
```

Setting the diagonal of the precision to 0 to more easily visualize the extradiagonal terms:
```python
p.assign_add(-tf.linalg.diag(tf.linalg.diag_part(p)))
```

Plotting setup:
```python
import matplotlib.pyplot as plt
import matplotlib.colors as colors
fig, axs = plt.subplots(2,2)
bounds = tf.linspace(-0.05, 0.05, 12)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
cmap = 'RdBu_r'
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
```

Plotting the reshaped rows of the learned precision matrix:
```python
def plot_m(ax, k):
    ax.imshow(tf.reshape(p[k,:], (28, 28)), cmap='RdBu_r', norm=norm)
    ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
plot_m(axs[0,0], 28 * 20 + 18)
plot_m(axs[0,1], 28 * 14 + 14)
plot_m(axs[1,0], 28 * 8 + 16)
plot_m(axs[1,1], 28 * 8 + 10)
```

Plotting the colorbar:
```python
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(sm, cax=cbar_ax, ticks=[-0.05, 0, 0.05])
cbar_ax.text(-0.3, 0.5, 'weight', horizontalalignment='center',
                verticalalignment='center', rotation='vertical',
                transform=cbar_ax.transAxes)
```

```python
>>> plt.show()
```    

<a href="https://ibb.co/ZJZfcFD"><img src="https://i.ibb.co/FWPh8yT/pwpevis.png" alt="pwpevis" border="0"></a>