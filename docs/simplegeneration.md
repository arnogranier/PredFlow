---
title: MNIST generation
tagline: A simple example of using predictive coding to generate MNIST digits
header-includes:
  - \usepackage{algorithm2e}
---

Here we will show how to train a simple multilayer perceptron with predictive coding to generate samples of MNIST digits.

Import tensorflow:
```python
import os ; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
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

\begin{algorithm}[H]
\DontPrintSemicolon
\SetAlgoLined
\KwResult{Write here the result}
\SetKwInOut{Input}{Input}\SetKwInOut{Output}{Output}
\Input{Write here the input}
\Output{Write here the output}
\BlankLine
\While{While condition}{
    instructions\;
    \eIf{condition}{
        instructions1\;
        instructions2\;
    }{
        instructions3\;
    }
}
\caption{While loop with If/Else condition}
\end{algorithm} 

Define a step of learning for the supervised generative predictive coding algorithm with weights `w`:
1. Clamp the top layer to the target and the bottom layer to the data
2. Initialize the hidden layers to the predictions $$W_if(r_{i+1})$$
3. Run an inference loop: <br>
do T times <br>
&nbsp;&nbsp;for all hidden layers <br>
&nbsp;&nbsp;&nbsp;&nbsp;$$e_i = r_i - W_if(r_{i+1})$$ <br>
&nbsp;&nbsp;&nbsp;&nbsp;$$r_i \mathrel{+}= ir * (-e_i + {W_i}^Te_i \odot f'(r_i))$$ <br>
4. Run a weight update step: <br>
for all weight matrices <br>
&nbsp;&nbsp;$$W_i \mathrel{+}= lr * (e_i \otimes f(r_{i+1}))$$

```python
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
            r[i] += tf.scalar_mul(ir, -e[i] + tf.matmul(w[i-1], e[i-1],
                                  transpose_a=True) * df(r[i]))
    
    # Learning
    for i in range(N):
        e[i] = tf.subtract(r[i], tf.matmul(w[i], f(r[i+1])))
        w[i].assign_add(tf.scalar_mul(lr, reduce_batch_outer(e[i], f(r[i+1]))))
```

Define an inference loop for the supervised generative predictive coding algorithm with weights `w`:
1. Clamp the top layer to the target
2. Initialize the hidden and bottom layers to the predictions $$W_if(r_{i+1})$$
3. Run an inference loop: <br>
do T times <br>
&nbsp;&nbsp;for all hidden layers <br>
&nbsp;&nbsp;&nbsp;&nbsp;$$e_i = r_i - W_if(r_{i+1})$$ <br>
&nbsp;&nbsp;&nbsp;&nbsp;$$r_i \mathrel{+}= ir * (-e_i + {W_i}^Te_i \odot f'(r_i))$$ <br>
&nbsp;&nbsp;# Bottom layer <br>
&nbsp;&nbsp;$$e_0 = r_0 - W_0f(r_{1})$$ <br>
&nbsp;&nbsp;$$r_0 \mathrel{+}= ir * (-e_0)$$ <br>
4. Return the infered representations
```python
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
            r[i] += tf.scalar_mul(ir, -e[i] + tf.matmul(w[i-1], e[i-1],
                                  transpose_a=True) * df(r[i]))
        r[0] += tf.scalar_mul(ir, -e[0])
    
    return r
```

Hyperparamaters: 
```python
batch_size = 50
mlp_architecture = [784, 256, 64, 10]
n_epochs = 5
w_init_std = 0.001
```

Load the MNIST dataset: 
```python
import tensorflow_datasets as tfds
def preprocess(image, label): 
    return (tf.reshape(tf.cast(image, tf.float32), [784, 1]) / 255.,
            tf.cast(tf.expand_dims(tf.one_hot(label, 10), -1), tf.float32))
ds = tfds.load('mnist', split='train', as_supervised=True)
ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.cache().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
```

Initialize the MLP weights: 
```python
w = [tf.Variable(tf.random.normal([s1, s2], stddev=w_init_std))
     for (s1, s2) in zip(mlp_architecture[:-1], mlp_architecture[1:])]
```

Training loop: 
```python
for epoch in range(n_epochs):
    ds.shuffle(60000)
    for image, target in ds:
        learn(w, image, target)
```

Generation of digits: 
```python
targets = tf.constant(tf.expand_dims(tf.eye(10), -1))
l = generate(w, targets)
```

Plot the generated digits: 
```python
import matplotlib.pyplot as plt
fig, _ = plt.subplots(2,5)
for i, ax in enumerate(fig.axes):
    ax.imshow(tf.reshape(l[0][i,:,:], (28,28)), cmap="Greys")
    ax.axis("off")
plt.tight_layout()
plt.show()
```

<a href="https://ibb.co/YQ6Dfd5"><img src="https://i.ibb.co/h9kX2dG/generation.png" alt="generation" border="0" height=240 width=320></a>