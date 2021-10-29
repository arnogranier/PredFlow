import tensorflow as tf
import tensorflow_datasets as tfds
from tf_utils import one_hot_pred_accuracy

def load_mnist(batch_size=50):    
    def preprocess(image, label):
        return tf.reshape(tf.cast(image, tf.float32), [784, 1]) / 255., tf.cast(tf.expand_dims(tf.one_hot(label, 10), -1), tf.float32)
    ds = tfds.load('mnist', split='train', as_supervised=True, shuffle_files=True).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
    tds = tfds.load('mnist', split='test', as_supervised=True).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(10000).cache().prefetch(tf.data.experimental.AUTOTUNE)
    
    return ds, tds