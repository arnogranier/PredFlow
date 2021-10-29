import tensorflow as tf
from tf_utils import *


def inference_step(e, r, w, ir, f, df, update_last=True):
    """[summary]

    Args:
        e (list of Tensors of float32): [description]
        r (list of Tensors of float32): [description]
        w (list of Tensors of float32): [description]
        ir (float32): [description]
        f (function): [description]
        df (function): [description]
        update_last (bool, optional): [description]. Defaults to True.
    """
    N = len(w)
    with tf.name_scope("PredictionErrorComputation"):
        for i in range(N):
            e[i] = tf.subtract(r[i], tf.matmul(w[i], f(r[i+1])))
    with tf.name_scope("RepresentationUpdate"):
        for i in range(1, N):
            r[i] += tf.scalar_mul(ir, tf.subtract(tf.matmul(w[i-1], e[i-1], transpose_a=True) * df(r[i]), e[i]))
        if update_last:
            r[N] += tf.scalar_mul(ir, tf.matmul(w[N-1], e[N-1], transpose_a=True) * df(r[N]))
            
def weight_update(w, e, r, lr, f):
    """[summary]

    Args:
        w (list of Tensors of float32): [description]
        e ([type]): [description]
        r ([type]): [description]
        lr ([type]): [description]
        f ([type]): [description]
    """
    
    for i in range(len(w)):
        w[i].assign_add(tf.scalar_mul(lr, reduced_batched_outer_product(e[i], f(r[i+1]))))

@tf.function
def learn(weights, image, target, ir=0.1, lr=0.0003, T=40, f=relu, df=d_relu):
    """[summary]

    Args:
        weights ([type]): [description]
        image ([type]): [description]
        target ([type]): [description]
        ir (float, optional): [description]. Defaults to 0.1.
        lr (float, optional): [description]. Defaults to 0.0003.
        T (int, optional): [description]. Defaults to 40.
        f ([type], optional): [description]. Defaults to relu.
        df ([type], optional): [description]. Defaults to d_relu.
    """
    N = len(weights)
    with tf.name_scope("Initialization"):
        representations = [target, ]
        for i in range(1,N):
            representations.insert(0, tf.matmul(weights[N-i], representations[0]))
        representations.insert(0, image)
        errors = [tf.zeros(tf.shape(representations[i])) for i in range(N)]

    with tf.name_scope("InferenceLoop"):
        for _ in range(T):
            with tf.name_scope("InferenceStep"):
                inference_step(errors, representations, weights, ir, f, df, update_last=False)
    with tf.name_scope("UpdateWeights"):
        weight_update(weights, errors, representations, lr, f)

@tf.function
def infer(weights, image, ir=0.01, T=200, f=relu, df=d_relu):
    """[summary]

    Args:
        weights ([type]): [description]
        image ([type]): [description]
        ir (float, optional): [description]. Defaults to 0.01.
        T (int, optional): [description]. Defaults to 200.
        f ([type], optional): [description]. Defaults to relu.
        df ([type], optional): [description]. Defaults to d_relu.

    Returns:
        [type]: [description]
    """
    
    N = len(weights)
    with tf.name_scope("Initialization"):
        representations = [image, ]
        for i in range(N):
            representations.append(tf.matmul(weights[i], representations[i], transpose_a=True)/10.)
        errors = [tf.zeros(tf.shape(representations[i])) for i in range(N)]
    with tf.name_scope("InferenceLoop"):
        for _ in range(T):
            with tf.name_scope("InferenceStep"):
                inference_step(errors, representations, weights, ir, f, df)
    return representations[1:]