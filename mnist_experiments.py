import os ; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf

from datasets_utils import load_mnist
from tf_utils import load_tensorboard_graph, mlp, one_hot_pred_accuracy
import supervised_autodiff_pc as pc
import time

if __name__ == "__main__":

    # Load MNIST dataset
    train_dataset, test_dataset = load_mnist(batch_size=100)
    
    # MLP model
    reverse = True
    model = mlp(784, 256, 64, 10, reversed_flow=reverse)
    
    # Train
    start = time.perf_counter()
    for epoch in range(2):
        train_dataset.shuffle(60000)
        for (image, target) in train_dataset:
            pc.learn(model, tf.constant(image), tf.constant(target), ir=tf.constant(.05),
                    lr=tf.constant(.005), T=40, predictions_flow_upward=not reverse)
    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)

    # Infer test set and compute accuracy
    
    (test_images, test_targets) = test_dataset.get_single_element()
    l = pc.infer(model, tf.constant(test_images), ir=tf.constant(.05), T=60,
                 predictions_flow_upward=not reverse, target_shape=list(tf.shape(test_targets).numpy()))
    tf.print(one_hot_pred_accuracy(test_targets, l[-1]))