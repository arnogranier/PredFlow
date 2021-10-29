from numpy import dtype
import tensorflow as tf
from pc_utils import *

@tf.function
def learn(model, image, target, ir=0.03, lr=0.0004, T=40, predictions_flow_upward=False):

    if predictions_flow_upward:
        representations = forward_initialize_representations(model, image, target)
    else:
        representations = backward_initialize_representations(model, target, image)
    parameters = [param for layer in model for param in layer.trainable_variables]
    
    with tf.name_scope("InferenceLoop"):
        for _ in range(T):
            with tf.name_scope("InferenceStep"):
                energy, gradients = energy_and_error(model, representations, parameters, predictions_flow_upward=predictions_flow_upward)
                inference_SGD_step(representations, ir, gradients.gradient(energy, representations), update_last=False)
            
    parameters_SGD_step(parameters, lr, gradients.gradient(energy, parameters))
    
    del gradients

@tf.function
def infer(model, image, ir=0.01, T=200, predictions_flow_upward=False, target_shape=None):
    
    if predictions_flow_upward:
        representations = forward_initialize_representations(model, image)
    else:
        representations = random_initialize_representations(model, image)
        
    with tf.name_scope("InferenceLoop"):
        for _ in range(T):
            with tf.name_scope("InferenceStep"):
                energy, gradients = energy_and_error(model, representations, predictions_flow_upward=predictions_flow_upward)
                inference_SGD_step(representations, ir, gradients.gradient(energy, representations))
                
    del gradients
    
    return representations[1:]