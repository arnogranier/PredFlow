'''
[description]
'''


import tensorflow as tf
from pc_utils import *
from tf_utils import relu_derivate

@tf.function
def learn(weights, image, target, ir=0.1, lr=0.0003, T=40, f=tf.nn.relu, df=relu_derivate, predictions_flow_upward=False):
    """[summary]

    :param weights: [description]
    :type weights: list of 2d tf.Tensor
    :param image: [description]
    :type image: [type]
    :param target: [description]
    :type target: [type]
    :param ir: [description], defaults to 0.1
    :type ir: float, optional
    :param lr: [description], defaults to 0.0003
    :type lr: float, optional
    :param T: [description], defaults to 40
    :type T: int, optional
    :param f: [description], defaults to tf.nn.relu
    :type f: [type], optional
    :param df: [description], defaults to relu_derivate
    :type df: [type], optional
    :param predictions_flow_upward: [description], defaults to False
    :type predictions_flow_upward: bool, optional
    """
    
    N = len(weights)
    with tf.name_scope("Initialization"):
        with tf.name_scope("RepresentationsInitialization"):
            if predictions_flow_upward:
                representations = [image, ]
                for i in range(N-1):
                    representations.append(tf.matmul(weights[i], f(representations[-1])))
                representations.append(target)
            else:
                representations = [target, ]
                for i in reversed(range(1,N)):
                    representations.insert(0, tf.matmul(weights[i], f(representations[0])))
                representations.insert(0, image)
        with tf.name_scope("ErrorsInitialization"):
            errors = [tf.zeros(tf.shape(representations[i])) for i in range(N)]

    with tf.name_scope("InferenceLoop"):
        for _ in range(T):
            with tf.name_scope("InferenceStep"):
                if predictions_flow_upward:
                    inference_step_forward_predictions(errors, representations, weights, ir, f, df, update_last=False)
                else:
                    inference_step_backward_predictions(errors, representations, weights, ir, f, df, update_last=False)
                    
    if predictions_flow_upward:
        weight_update_forward_predictions(weights, errors, representations, lr, f)
    else:
        weight_update_backward_predictions(weights, errors, representations, lr, f)
        
@tf.function
def infer(weights, image, ir=0.01, T=200, f=tf.nn.relu, df=relu_derivate, predictions_flow_upward=False, target_shape=None):
    """[summary]

    :param weights: [description]
    :type weights: [type]
    :param image: [description]
    :type image: [type]
    :param ir: [description], defaults to 0.01
    :type ir: float, optional
    :param T: [description], defaults to 200
    :type T: int, optional
    :param f: [description], defaults to tf.nn.relu
    :type f: [type], optional
    :param df: [description], defaults to relu_derivate
    :type df: [type], optional
    :param predictions_flow_upward: [description], defaults to False
    :type predictions_flow_upward: bool, optional
    :param target_shape: [description], defaults to None
    :type target_shape: [type], optional
    :return: [description]
    :rtype: [type]
    """
    
    N = len(weights)
    with tf.name_scope("Initialization"):
        if predictions_flow_upward:
            representations = [image, ]
            for i in range(N):
                representations.append(tf.matmul(weights[i], f(representations[-1])))
        else:
            representations = [tf.zeros(target_shape)+tf.constant(.0001),]
            for i in reversed(range(1,N)):
                representations.insert(0, tf.zeros(tf.shape(tf.matmul(weights[i], representations[0])))+tf.constant(.0001))
            representations.insert(0, image)
            
        errors = [tf.zeros(tf.shape(representations[i])) for i in range(N)]
        
    with tf.name_scope("InferenceLoop"):
        for _ in range(T):
            with tf.name_scope("InferenceStep"):
                if predictions_flow_upward:
                    inference_step_forward_predictions(errors, representations, weights, ir, f, df)
                else:
                    inference_step_backward_predictions(errors, representations, weights, ir, f, df)
                    
    return representations[1:]