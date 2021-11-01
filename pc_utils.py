'''
[description]
'''


import tensorflow as tf 
from tf_utils import reduced_batched_outer_product

def inference_SGD_step(r, ir, g, update_last=True):
    """[summary]

    :param r: [description]
    :type r: list of 3d tf.Tensor of float32
    :param ir: [description]
    :type ir: float32
    :param g: [description]
    :type g: tf.GradientTape
    :param update_last: [description], defaults to True
    :type update_last: bool, optional
    """
    
    N = len(r) - 1
    with tf.name_scope("RepresentationUpdate"):
        for i in range(1, N):
            r[i] -= tf.scalar_mul(ir, g[i])
        if update_last:
            r[N] -= tf.scalar_mul(ir, g[N])
    
def parameters_SGD_step(theta, lr, g):
    """[summary]

    :param theta: [description]
    :type theta: list of variable tf.Tensor of float32
    :param lr: [description]
    :type lr: float32
    :param g: [description]
    :type g: tf.GradientTape
    """
    
    with tf.name_scope("ParametersUpdate"):
        for i in range(len(theta)):
            theta[i].assign_add(tf.scalar_mul(lr, -g[i]))
    
def energy_and_error(model, r, theta=[], predictions_flow_upward=False):
    """[summary]

    :param model: [description]
    :type model: list of tf_utils.Dense or tf_utils.BiasedDense
    :param r: [description]
    :type r: list of 3d tf.Tensor of float32
    :param theta: [description], defaults to []
    :type theta: list of variable tf.Tensor of float32, optional
    :param predictions_flow_upward: [description], defaults to False
    :type predictions_flow_upward: bool, optional
    :return: [description]
    :rtype: float32, tf.GradientTape
    """
    
    with tf.name_scope("EnergyComputation"):
        F = tf.zeros(())
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(r+theta)
            for i in range(len(model)):
                if predictions_flow_upward:
                    F += 0.5 * tf.reduce_sum(tf.square(tf.subtract(r[i+1], model[i](r[i]))), 1)
                else:
                    F += 0.5 * tf.reduce_sum(tf.square(tf.subtract(r[i], model[i](r[i+1]))), 1)
        return F, tape

def forward_initialize_representations(model, image, target=None):
    """[summary]

    :param model: [description]
    :type model: list of tf_utils.Dense or tf_utils.BiasedDense
    :param image: [description]
    :type image: 3d tf.Tensor of float32
    :param target: [description], defaults to None
    :type target: 3d tf.Tensor of float32, optional
    :return: [description]
    :rtype: list of 3d tf.Tensor of float32
    """
    
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
    
def backward_initialize_representations(model, target, image=None):
    """[summary]

    :param model: [description]
    :type model: list of tf_utils.Dense or tf_utils.BiasedDense
    :param target: [description]
    :type target: 3d tf.Tensor of float32
    :param image: [description], defaults to None
    :type image: 3d tf.Tensor of float32, optional
    :return: [description]
    :rtype: list of 3d tf.Tensor of float32
    """
    
    with tf.name_scope("Initialization"):
        N = len(model)
        representations = [target,]
        for i in reversed(range(1, N)):
            representations.insert(0, model[i](representations[0]))
        if image is not None:
            representations.insert(0, image)
        else:
            representations.insert(0, model[0](representations[0]))
        return representations
    
def random_initialize_representations(model, image, stddev=0.001, predictions_flow_upward=False, target_shape=None):
    """[summary]

    :param model: [description]
    :type model: list of tf_utils.Dense or tf_utils.BiasedDense
    :param image: [description]
    :type image: 3d tf.Tensor of float32
    :param stddev: [description], defaults to 0.001
    :type stddev: float, optional
    :param predictions_flow_upward: [description], defaults to False
    :type predictions_flow_upward: bool, optional
    :param target_shape: [description], defaults to None
    :type target_shape: tf.Tensor of int32, optional
    :return: [description]
    :rtype: list of 3d tf.Tensor of float32
    """
    
    with tf.name_scope("Initialization"):
        N = len(model)
        if predictions_flow_upward:
            representations = [image,]
            for i in range(N):
                representations.append(tf.random.normal(tf.shape(model[i](representations[-1])), stddev=stddev))
        else:
            representations = [tf.random.normal(target_shape, stddev=stddev),]
            for i in reversed(range(1, N)):
                representations.insert(0, tf.random.normal(tf.shape(model[i](representations[0])),stddev=stddev))
            representations.insert(0, image)
        return representations
    
def zero_initialize_representations(model, image, predictions_flow_upward=False, target_shape=None, bias=tf.constant(0.)):
    """[summary]

    :param model: [description]
    :type model: list of tf_utils.Dense or tf_utils.BiasedDense
    :param image: [description]
    :type image: 3d tf.Tensor of float32
    :param predictions_flow_upward: [description], defaults to False
    :type predictions_flow_upward: bool, optional
    :param target_shape: [description], defaults to None
    :type target_shape: tf.Tensor of int32, optional
    :param bias: [description], defaults to tf.constant(0.)
    :type bias: float32, optional
    :return: [description]
    :rtype: list of 3d tf.Tensor of float32
    """
    
    with tf.name_scope("Initialization"):
        N = len(model)
        if predictions_flow_upward:
            representations = [image,]
            for i in range(N):
                representations.append(bias+tf.zeros(tf.shape(model[i](representations[-1]))))
        else:
            representations = [tf.zeros(target_shape)+bias,]
            for i in reversed(range(1, N)):
                representations.insert(0, tf.zeros(tf.shape(model[i](representations[0])))+bias)
            representations.insert(0, image)
        return representations
    
def inference_step_backward_predictions(e, r, w, ir, f, df, update_last=True):
    """[summary]

    :param e: [description]
    :type e: list of 3d tf.Tensor of float32
    :param r: [description]
    :type r: list of 3d tf.Tensor of float32
    :param w: [description]
    :type w: list of 2d tf.Tensor of float32
    :param ir: [description]
    :type ir: float32
    :param f: [description]
    :type f: function
    :param df: [description]
    :type df: function
    :param update_last: [description], defaults to True
    :type update_last: bool, optional
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
            
def weight_update_backward_predictions(w, e, r, lr, f):
    """[summary]

    :param w: [description]
    :type w: list of 2d tf.Tensor of float32
    :param e: [description]
    :type e: list of 3d tf.Tensor of float32
    :param r: [description]
    :type r: list of 3d tf.Tensor of float32
    :param lr: [description]
    :type lr: float32
    :param f: [description]
    :type f: function
    """
    
    with tf.name_scope("WeightUpdate"):
        for i in range(len(w)):
            w[i].assign_add(tf.scalar_mul(lr, reduced_batched_outer_product(e[i], f(r[i+1]))))

def inference_step_forward_predictions(e, r, w, ir, f, df, update_last=True):
    """[summary]

    :param e: [description]
    :type e: list of 3d tf.Tensor of float32
    :param r: [description]
    :type r: list of 3d tf.Tensor of float32
    :param w: [description]
    :type w: list of 2d tf.Tensor of float32
    :param ir: [description]
    :type ir: float32
    :param f: [description]
    :type f: float32
    :param df: [description]
    :type df: function
    :param update_last: [description], defaults to True
    :type update_last: bool, optional
    """
    
    N = len(w)
    with tf.name_scope("PredictionErrorComputation"):
        for i in range(N):
            e[i] = tf.subtract(r[i+1], tf.matmul(w[i], f(r[i])))
    with tf.name_scope("RepresentationUpdate"):
        for i in range(1,N):
            r[i] += tf.scalar_mul(ir, tf.subtract(tf.matmul(w[i], e[i], transpose_a=True) * df(r[i]), e[i-1]))
        if update_last:
            r[N] -= tf.scalar_mul(ir, e[N-1])
            
def weight_update_forward_predictions(w, e, r, lr, f):
    """[summary]

    :param w: [description]
    :type w: list of 2d tf.Tensor of float32
    :param e: [description]
    :type e: list of 3d tf.Tensor of float32
    :param r: [description]
    :type r: list of 3d tf.Tensor of float32
    :param lr: [description]
    :type lr: float32
    :param f: [description]
    :type f: function
    """
    
    with tf.name_scope("WeightUpdate"):
        for i in range(len(w)):
            w[i].assign_add(tf.scalar_mul(lr, reduced_batched_outer_product(e[i], f(r[i]))))
