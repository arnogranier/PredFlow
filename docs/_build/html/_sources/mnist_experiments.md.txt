# Train a MLP classifier on MNIST

Here we will show how to train a simple MLP classifier on MNIST with predictive 
coding, using the version of predictive coding where predictions
flow up the network.


```python
from os import environ ; environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from time import perf_counter
```

```python
from predflow.datasets_utils import load_mnist
from predflow.tf_utils import mlp, one_hot_pred_accuracy
import predflow.supervised_fullyconnected_stricthierarchy_explicit_pc as pc
```

```python
# Load MNIST dataset
train_dataset, test_dataset = load_mnist(batch_size=100)
```

```python
# MLP model
W = mlp(784, 256, 64, 10, only_return_variables=True)
```

```python
# Train
start = perf_counter()
for epoch in range(10):
    train_dataset.shuffle(60000)
    for image, target in train_dataset:
        pc.learn(W, tf.constant(image), tf.constant(target), ir=tf.constant(.05),
                 lr=tf.constant(.01), T=10, predictions_flow_upward=True)
elapsed = perf_counter() - start
```

```python
>>> elapsed
19.691 seconds
```

```python
# Infer test set and compute accuracy
(test_images, test_targets) = test_dataset.get_single_element()
l = pc.infer(W, tf.constant(test_images), predictions_flow_upward=True, T=0,
             target_shape=list(tf.shape(test_targets).numpy()))
```

```python
>>> one_hot_pred_accuracy(test_targets, l[-1])
0.9778
```
