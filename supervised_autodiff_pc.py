from numpy import dtype
import tensorflow as tf
from pc_utils import *

@tf.function
def learn(model, image, target, ir=0.1, lr=0.001, T=40, predictions_flow_upward=False):

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
def infer(model, image, ir=0.025, T=200, predictions_flow_upward=False):
    
    if predictions_flow_upward:
        representations = forward_initialize_representations(model, image)
    else:
        representations = random_initialize_representations(model, image)
        
    with tf.name_scope("InferenceLoop"):
        for _ in range(T):
            with tf.name_scope("InferenceStep"):
                energy, autodiff = energy_and_error(model, representations, predictions_flow_upward=predictions_flow_upward)
                inference_SGD_step(representations, ir, autodiff.gradient(energy, representations))
                
    del autodiff
    
    return representations[1:]