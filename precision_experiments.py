import os ; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf

from datasets_utils import load_mnist
from tf_utils import load_tensorboard_graph, mlp, one_hot_pred_accuracy
import precision_modulated_supervised_explicit_pc as pc
import time
if __name__ == "__main__":

    # Load MNIST dataset
    train_dataset, test_dataset = load_mnist(batch_size=100)
    
    # MLP model
    reverse = False
    (w, P) = mlp(784, 256, 64, 10, reversed_flow=reverse, precision_modulated=True, only_return_variables=1)

    # Train
    start = time.perf_counter()
    for epoch in range(30):
        train_dataset.shuffle(60000)
        for (image, target) in train_dataset:  
            pc.learn(w, P, tf.constant(image), tf.constant(target), ir=tf.constant(.005),
                     pr=tf.constant(.002), lr=tf.constant(.005), T=40, predictions_flow_upward=not reverse, 
                     diagonal=False)
        print(P[-1])
    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)

    # Infer test set and compute accuracy
    (test_images, test_targets) = test_dataset.get_single_element()
    l = pc.infer(w,P, tf.constant(test_images), ir=tf.constant(.05), T=10,
                 predictions_flow_upward=not reverse, target_shape=tf.shape(test_targets))
    tf.print(one_hot_pred_accuracy(test_targets, l[-1]))
