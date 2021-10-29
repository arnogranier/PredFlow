from numpy import dtype
import tensorflow as tf
from tf_utils import *


def inference_step(r, ir, g, update_last=True):
    N = len(r) - 1
    with tf.name_scope("RepresentationUpdate"):
        for i in range(1, N):
            r[i] -= tf.scalar_mul(ir, g[i])
        if update_last:
            r[N] -= tf.scalar_mul(ir, g[N])
    
def parameters_update(theta, lr, g):
    for i in range(len(theta)):
        theta[i].assign_add(tf.scalar_mul(lr, -g[i]))
    
def energy_and_error(w, r, theta=[]):
    with tf.name_scope("ErrorEnergyComputation"):
        F = tf.zeros(())
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
            g.watch(r+theta)
            for i in range(len(w)):
                F += 0.5 * tf.reduce_sum(tf.square(tf.subtract(r[i+1], w[i](r[i]))), 1)
        return F, g

def forward_initialize_representations(model, image, target=None):
    with tf.name_scope("Initialization"):
        N = len(model)
        representations = [image,]
        for i in range(N-1):
            representations.append(model[i](representations[-1]))
        if target is not None:
            representations.append(target)
        else:
            representations.append(model[-1](representations[-1]))
        return representations

@tf.function
def learn(model, image, target, ir=0.03, lr=0.0004, T=40):

    representations = forward_initialize_representations(model, image, target)
    parameters = [param for layer in model for param in layer.trainable_variables]
    
    with tf.name_scope("InferenceLoop"):
        for _ in range(T):
            with tf.name_scope("InferenceStep"):
                energy, gradients = energy_and_error(model, representations, parameters)
                inference_step(representations, ir, gradients.gradient(energy, representations), update_last=False)
                
    with tf.name_scope("ParametersUpdate"):
        parameters_update(parameters, lr, gradients.gradient(energy, parameters))
    
    del gradients

@tf.function
def infer(model, image, ir=0.01, T=200):
    
    representations = forward_initialize_representations(model, image)
        
    with tf.name_scope("InferenceLoop"):
        for _ in range(T):
            with tf.name_scope("InferenceStep"):
                energy, gradients = energy_and_error(model, representations)
                inference_step(representations, ir, gradients.gradient(energy, representations))
                
    del gradients
    
    return representations[1:]