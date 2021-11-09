# Train a MLP classifier on MNIST

Here we will show how to train a simple MLP classifier on MNIST with predictive 
coding. Here we use the version of predictive coding where predictions
flow up the network.

```python
import tensorflow as tf

import supervised_autodiff_pc as pc

from datasets_utils import load_mnist
from tf_utils import mlp, one_hot_pred_accuracy

from time import perf_counter
```

```python
# MLP model
model = mlp(784, 256, 64, 10)
```

```python
# Load MNIST dataset
train_dataset, test_dataset = load_mnist(batch_size=32)
```

```python
# Train
start = time.perf_counter()
for epoch in range(2):
    train_dataset.shuffle(60000)
    for (image, target) in train_dataset:
        # Calling pc.learn on minibatch
        pc.learn(model, tf.constant(image), tf.constant(target), ir=tf.constant(.1),
                lr=tf.constant(.005), T=20, predictions_flow_upward=True)
elapsed = time.perf_counter() - start
```

```python
>>> elapsed
18.563
```

```python
# Infer test set and compute accuracy
(test_images, test_targets) = test_dataset.get_single_element()
l = pc.infer(model, tf.constant(test_images), ir=tf.constant(.025),
             predictions_flow_upward=True)
```

```python
>>> one_hot_pred_accuracy(test_targets, l[-1])
0.9615
```