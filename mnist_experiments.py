import tensorflow as tf

from datasets_utils import load_mnist
from tf_utils import load_tensorboard_graph, mlp, one_hot_pred_accuracy
import supervised_fullyconnected_stricthierarchy_explicit_pc as pc
import time

if __name__ == "__main__":

    # Load MNIST dataset
    train_dataset, test_dataset = load_mnist(batch_size=32)

    # MLP model
    model = mlp(784, 256, 64, 10, only_return_weights=True)
    # Train
    start = time.perf_counter()
    for epoch in range(1):
        #train_dataset.shuffle(60000)
        for (image, target) in train_dataset:
            load_tensorboard_graph('logs', pc.learn, [model, tf.constant(image), tf.constant(target)], 'learn_trace', kwargs={'ir':tf.constant(.1),
                    'lr':tf.constant(.005), 'T':5, 'predictions_flow_upward':True})
            break
    """elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)

    # Infer test set and compute accuracy
    (test_images, test_targets) = test_dataset.get_single_element()
    l = pc.infer(model, tf.constant(test_images), ir=tf.constant(.025),
                 predictions_flow_upward=True, target_shape=tf.shape(test_targets))
    tf.print(one_hot_pred_accuracy(test_targets, l[-1]))"""