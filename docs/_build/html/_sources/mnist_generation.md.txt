# Generate MNIST digits

Here we will show how to train a simple multilayer perceptron with predictive coding to generate samples of MNIST digits.

```python
from os import environ ; environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
import matplotlib.pyplot as plt
```

```python
from predflow.datasets_utils import load_mnist
from predflow.tf_utils import mlp
import predflow.supervised_fullyconnected_stricthierarchy_explicit_pc as pc
```

```python
# Load MNIST dataset
train_dataset, test_dataset = load_mnist(batch_size=100)
```

```python
# MLP model
W = mlp(784, 256, 64, 10, reversed_flow=True, only_return_variables=True)
```

```python
# Train
for epoch in range(10):
    train_dataset.shuffle(60000)
    for image, target in train_dataset:
        pc.learn(W, tf.constant(image), tf.constant(target), ir=tf.constant(.05),
                 lr=tf.constant(.002), T=30)
```

```python
# Generate sensory representations for each one-hot encoded label
targets = tf.expand_dims(tf.eye(10), -1)
l = pc.generate(W, tf.constant(targets), ir=tf.constant(.05), T=30)
```

```python
# Plots
fig, _ = plt.subplots(2,5)
for i, ax in enumerate(fig.axes):
    ax.imshow(tf.reshape(l[0][i,:,:], (28,28)), cmap="Greys")
    ax.axis("off")
plt.tight_layout()
plt.show()
```

<a href="https://ibb.co/YQ6Dfd5"><img src="https://i.ibb.co/h9kX2dG/generation.png" alt="generation" border="0" height=240 width=320></a>