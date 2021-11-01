'''
[description]
'''


import tensorflow as tf
import tensorflow_datasets as tfds

def load_mnist(batch_size=50):
    """[summary]

    :param batch_size: [description], defaults to 50
    :type batch_size: int, optional
    """
    
    def preprocess(image, label):
        return tf.reshape(tf.cast(image, tf.float32), [784, 1]) / 255., tf.cast(tf.expand_dims(tf.one_hot(label, 10), -1), tf.float32)
    ds = tfds.load('mnist', split='train', as_supervised=True, shuffle_files=True).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).cache().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    tds = tfds.load('mnist', split='test', as_supervised=True).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).cache().batch(10000).prefetch(tf.data.experimental.AUTOTUNE)
    
    return ds, tds