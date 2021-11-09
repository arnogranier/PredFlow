import tensorflow as tf

from datasets_utils import load_mnist
from tf_utils import load_tensorboard_graph, mlp, one_hot_pred_accuracy
import supervised_autodiff_pc as pc
import time

if __name__ == "__main__":

    # Load MNIST dataset
    train_dataset, test_dataset = load_mnist(batch_size=32)

    # MLP model
    model = mlp(784, 400, 400, 10, reversed_flow=True)
    # Train
    start = time.perf_counter()
    for epoch in range(2):
        train_dataset.shuffle(60000)
        for (image, target) in train_dataset:
            pc.learn(model, tf.constant(image), tf.constant(target), ir=tf.constant(.1),
                    lr=tf.constant(.01), T=20, predictions_flow_upward=False)
    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)

    # Infer test set and compute accuracy
    (test_images, test_targets) = test_dataset.get_single_element()
    l = pc.infer(model, tf.constant(test_images), ir=tf.constant(.05), T=60,
                 predictions_flow_upward=False, target_shape=tf.shape(test_targets))
    tf.print(one_hot_pred_accuracy(test_targets, l[-1]))