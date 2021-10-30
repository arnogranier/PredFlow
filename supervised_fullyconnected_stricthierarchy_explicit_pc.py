import tensorflow as tf
from pc_utils import *
from tf_utils import relu_derivate

@tf.function
def learn(weights, image, target, ir=0.1, lr=0.0003, T=40, f=tf.nn.relu, df=relu_derivate, predictions_flow_upward=False):
    """[summary]

    Args:
        weights ([type]): [description]
        image ([type]): [description]
        target ([type]): [description]
        ir (float, optional): [description]. Defaults to 0.1.
        lr (float, optional): [description]. Defaults to 0.0003.
        T (int, optional): [description]. Defaults to 40.
        f ([type], optional): [description]. Defaults to tf.nn.relu.
        df ([type], optional): [description]. Defaults to relu_derivate.
        predictions_flow_upward (bool, optional): [description]. Defaults to False.
    """
    N = len(weights)
    with tf.name_scope("Initialization"):
        if predictions_flow_upward:
            representations = [image, ]
            for i in range(N-1):
                representations.append(tf.matmul(weights[i], representations[-1]))
            representations.append(target)
        else:
            representations = [target, ]
            for i in range(1,N):
                representations.insert(0, tf.matmul(weights[N-i], representations[0]))
            representations.insert(0, image)
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
def infer(weights, image, ir=0.01, T=200, f=tf.nn.relu, df=relu_derivate, predictions_flow_upward=False):
    
    N = len(weights)
    with tf.name_scope("Initialization"):
        representations = [image, ]
        for i in range(N):
            if predictions_flow_upward:
                representations.append(tf.matmul(weights[i], representations[-1]))
            else:
                representations.append(tf.zeros(tf.shape(tf.matmul(weights[i], representations[-1]))))
        errors = [tf.zeros(tf.shape(representations[i])) for i in range(N)]
        
    with tf.name_scope("InferenceLoop"):
        for _ in range(T):
            with tf.name_scope("InferenceStep"):
                if predictions_flow_upward:
                    inference_step_forward_predictions(errors, representations, weights, ir, f, df, update_last=False)
                else:
                    inference_step_backward_predictions(errors, representations, weights, ir, f, df, update_last=False)
                    
    return representations[1:]