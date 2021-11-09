import tensorflow as tf

from datasets_utils import load_mnist
from tf_utils import load_tensorboard_graph, mlp, one_hot_pred_accuracy
import precision_modulated_supervised_explicit_pc as pc
import time
import matplotlib.pyplot as plt 

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


