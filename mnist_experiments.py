import tensorflow as tf

from datasets_utils import load_mnist
from tf_utils import load_tensorboard_graph, mlp, one_hot_pred_accuracy
import supervised_fullyconnected_stricthierarchy_explicit_pc as pc
import time

if __name__ == "__main__":

    # Load MNIST dataset
    train_dataset, test_dataset = load_mnist(batch_size=32)

    # Fully connected strict hierarchy weights
    model = mlp(784, 128, 10, only_return_weights=True)

    # Train
    start = time.perf_counter()
    for epoch in range(2):
        train_dataset.shuffle(60000)
        for (image, target) in train_dataset:
            pc.learn(model, tf.constant(image), tf.constant(target), ir=tf.constant(.1),
                    lr=tf.constant(.001), T=20, predictions_flow_upward=True)
    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)

    # Infer test set and compute accuracy
    for (image, target) in test_dataset:
        l = pc.infer(model, tf.constant(image), ir=tf.constant(.01),
                    predictions_flow_upward=True, target_shape=tf.shape(target))
        tf.print(one_hot_pred_accuracy(target, l[-1]))