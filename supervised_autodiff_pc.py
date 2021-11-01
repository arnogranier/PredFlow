'''
[description]
'''

import tensorflow as tf
from pc_utils import *

@tf.function
def learn(model, image, target, ir=0.1, lr=0.001, T=40, predictions_flow_upward=False):
    """[summary]

    :param model: [description]
    :type model: [type]
    :param image: [description]
    :type image: [type]
    :param target: [description]
    :type target: [type]
    :param ir: [description], defaults to 0.1
    :type ir: float, optional
    :param lr: [description], defaults to 0.001
    :type lr: float, optional
    :param T: [description], defaults to 40
    :type T: int, optional
    :param predictions_flow_upward: [description], defaults to False
    :type predictions_flow_upward: bool, optional
    """
    
    if predictions_flow_upward:
        representations = forward_initialize_representations(model, image, target)
    else:
        representations = backward_initialize_representations(model, target, image)
    parameters = [param for layer in model for param in layer.trainable_variables]
    
    with tf.name_scope("InferenceLoop"):
        for _ in range(T):
            with tf.name_scope("InferenceStep"):
                energy, autodiff = energy_and_error(model, representations, parameters, predictions_flow_upward=predictions_flow_upward)
                inference_SGD_step(representations, ir, autodiff.gradient(energy, representations), update_last=False)
            
    parameters_SGD_step(parameters, lr, autodiff.gradient(energy, parameters))
    
    del autodiff

@tf.function
def infer(model, image, ir=0.025, T=200, predictions_flow_upward=False, target_shape=None):
    """[summary]

    :param model: [description]
    :type model: [type]
    :param image: [description]
    :type image: [type]
    :param ir: [description], defaults to 0.025
    :type ir: float, optional
    :param T: [description], defaults to 200
    :type T: int, optional
    :param predictions_flow_upward: [description], defaults to False
    :type predictions_flow_upward: bool, optional
    :param target_shape: [description], defaults to None
    :type target_shape: [type], optional
    :return: [description]
    :rtype: [type]
    """
    
    if predictions_flow_upward:
        representations = forward_initialize_representations(model, image)
    else:
        representations = zero_initialize_representations(model, image, predictions_flow_upward=predictions_flow_upward, target_shape=target_shape, bias=tf.constant(.0001))
        
    with tf.name_scope("InferenceLoop"):
        for _ in range(T):
            with tf.name_scope("InferenceStep"):
                energy, autodiff = energy_and_error(model, representations, predictions_flow_upward=predictions_flow_upward)
                inference_SGD_step(representations, ir, autodiff.gradient(energy, representations))
                
    del autodiff
    
    return representations[1:]