One of the advantage of tensorflow is that you can easily visualize the
computational graph that is actually running on your computer
using [tensorboard](https://www.tensorflow.org/tensorboard). 

Here is what the computational graph of the simple `learn` function from the [example of MNIST digits generation](simplegeneration.md) from a high level view:

<a href="https://ibb.co/WDgW16N"><img src="https://i.ibb.co/cbDcd1K/Screenshot-from-2022-03-19-19-30-51.png" alt="Screenshot-from-2022-03-19-19-30-51" border="0"></a>

Of course we can get to a finer level of detail, and inspect for example the initialization phase:
<a href="https://ibb.co/DzjJPNv"><img src="https://i.ibb.co/rFXjKgz/Screenshot-from-2022-03-19-19-31-49.png" alt="Screenshot-from-2022-03-19-19-31-49" border="0"></a>

We can see that the initialization consist of a sweep through the model, stopping before the first layer since its activity is clamped to data, and an initalization of prediction errors to zero.

Let us also further inspect the graph of the inference loop:
<a href="https://ibb.co/1mgy2JG"><img src="https://i.ibb.co/6WM9sJ0/Screenshot-from-2022-03-19-19-32-30.png" alt="Screenshot-from-2022-03-19-19-32-30" border="0"></a>

We can see that it is alternating sequentially between prediction error computation and representation update (for 5 timesteps here). We can further inspect prediction errors computation:
<a href="https://ibb.co/FgNZ4bk"><img src="https://i.ibb.co/LnbjdzD/Screenshot-from-2022-03-19-19-33-33.png" alt="Screenshot-from-2022-03-19-19-33-33" border="0" width="500"></a>

and remark that it indeed computes $$e_i = r_i - W_if(r_{i+1})$$ as expected for this model. 

A further inspection of representation update illustrates the computation $$r_i \mathrel{+}= ir * (-e_i + {W_{i-1}}^Te_{i-1} \odot f'(r_i))$$:
<a href="https://ibb.co/gPmPTWx"><img src="https://i.ibb.co/TbYb80s/Screenshot-from-2022-03-19-19-33-08.png" alt="Screenshot-from-2022-03-19-19-33-08" border="0" width="700"></a>

Finally we can inspect the weight update computational graph, illustrating the computation $$W_i \mathrel{+}= lr * (e_i \otimes f(r_{i+1}))$$
<a href="https://ibb.co/2kprRdw"><img src="https://i.ibb.co/XCGP6t9/Screenshot-from-2022-03-19-19-34-46.png" alt="Screenshot-from-2022-03-19-19-34-46" border="0" width="00"></a>

A particularly important feature of the three last core computational graphs is that operations for each layers and weight matrices are executed in parallel (nodes on the same horizontal level in tensorboard graphs are executed in parallel). This certainly illustrates an important property of predictive coding, namely that it is highly parallelizable across layers, since there is no need to backpropagate gradients (because representations and parameters update are based on _local_ prediction errors).

Here is the code generating this tensorboard graph. It is very similar to [this code](simplegeneration.md), but we added identifier using `tensorflow.name_scope` to specify the names of our high-level operations, and we only go through one step of learning since it is the only thing needed to elicit a trace of the algorithm for tensorboard to plot.

```python
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
    logdir = 'YOUR_DIR_NAME'
    trace_name = 'trace'

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

    # Tensorboard writer
    writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on()
    
    # Training for one step
    for image, target in ds:
        learn(w, tf.constant(image), tf.constant(target))
        break
    
    # Exporting trace
    with writer.as_default():
        tf.summary.trace_export(
            name=trace_name,
            step=0,
            profiler_outdir=logdir)
```