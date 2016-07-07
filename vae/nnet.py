import numpy as np
import tensorflow as tf

### constructing and composing layers

def make_layer(dot, activation):
    def layer(W, b):
        def apply(h):
            return activation(dot(h, W) + b)
        return apply
    return layer


def compose(layers):
    return reduce(lambda f,g: lambda h: g(f(h)), layers, lambda x: x)


### initialization

def init_tensor(shape, name=None):
    init = tf.random_normal(shape, stddev=.1)
    return tf.Variable(init, name=name)

def init_layer(shape, layer_name=""):
    return init_tensor(shape,    name="%s_W"%layer_name), \
           init_tensor([shape[1]], name="%s_b"%layer_name)


### tensorflow-backed layers

tanh_layer    = make_layer(tf.matmul, tf.tanh)
sigmoid_layer = make_layer(tf.matmul, tf.nn.sigmoid)
relu_layer    = make_layer(tf.matmul, tf.nn.relu)
linear_layer  = make_layer(tf.matmul, lambda x: x)


### numpy-backed layers
numpy_tanh_layer    = make_layer(np.dot, np.tanh)
numpy_sigmoid_layer = make_layer(np.dot, lambda x: 1./(1. + np.exp(-x)))
numpy_linear_layer  = make_layer(np.dot, lambda x: x)
