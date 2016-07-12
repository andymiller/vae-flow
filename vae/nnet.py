import numpy as np
import tensorflow as tf

### constructing and composing layers

def make_layer(dot, activation):
    def layer(W, b):
        def apply(h):
            return activation(dot(h, W) + b)
        return apply
    return layer


# TODO - a stochastic layer is really a composition of a deterministic layer 
# and adding some gaussian noise, h_L => stochastic => h_{L-1}: T(h_L) + G \eta
def make_stochastic_layer(dot, activation):
    def layer(G, b):
        eta_dim = G.get_shape()[1].value
        def apply(h):
            h0  = h.get_shape()[0].value
            eta = tf.random_normal(shape=[h0, eta_dim])
            return h + dot(eta, G)
        return apply
    return layer


def compose(layers):
    return reduce(lambda f,g: lambda h: g(f(h)), layers, lambda x: x)


### initialization

def init_tensor(shape, name=None):
    init = tf.truncated_normal(shape, stddev=.1, dtype=tf.float32)
    return tf.Variable(init, name=name, dtype=np.float32)

def init_layer(shape, layer, layer_name=""):
    if layer is not stochastic_tanh_layer:
      return init_tensor(shape,      name="%s_W"%layer_name), \
             init_tensor([shape[1]], name="%s_b"%layer_name)
    else:
      return init_stochastic_layer(shape, layer_name)

def init_stochastic_layer(shape, layer_name=""):
    return init_tensor(shape, name="%s_G"%layer_name), None


### tensorflow-backed layers

tanh_layer       = make_layer(tf.matmul, tf.tanh)
sigmoid_layer    = make_layer(tf.matmul, tf.nn.sigmoid)
relu_layer       = make_layer(tf.matmul, tf.nn.relu)
linear_layer     = make_layer(tf.matmul, lambda x: x)
stochastic_tanh_layer = make_stochastic_layer(tf.matmul, tf.tanh)


### numpy-backed layers
numpy_tanh_layer    = make_layer(np.dot, np.tanh)
numpy_sigmoid_layer = make_layer(np.dot, lambda x: 1./(1. + np.exp(-x)))
numpy_linear_layer  = make_layer(np.dot, lambda x: x)


### mlp-maker TODO FINISH OUTPUT
def make_mlp(layers, out_dim, out_layers=None):
    """
    Follows the convention: 
        Each layer in the MLP is specified (hidden-dim, layer-type)
          so a [(10, tanh), (20, tanh)] with outputsize 2
          implies a set of parameters
             W_1 (10 x 20), b_1 (20 x 1)
             W_2 (20 x 2),  b_2 (2 x 1)

    """
    #first construct all non-output layers
    hidden_dims    = [l[0] for l in layers]
    shapes         = zip(hidden_dims[:-1], hidden_dims[1:])
    hidden_nonlins = [l[1] for l in layers[:-1]]
    hidden_params  = [init_layer(shape, l, layer_name="%d"%i)
                      for i, (shape, l) in enumerate(zip(shapes, hidden_nonlins))]

    # construct (potentially) multi-output layer
    out_nonlins = layers[-1][1:]
    out_shape   = (hidden_dims[-1], out_dim)
    out_params  = [init_layer(out_shape, l, layer_name="out_%d"%i)
                   for i, l in enumerate(out_nonlins)]

    # string together hidden layers and output layer
    hmlp = compose(l(W, b) for (W, b), l in zip(hidden_params, hidden_nonlins))
    output = [l(W,b) for (W,b), l in zip(out_params, out_nonlins)]

    # TODO find a nicer way to output a list vs single
    if len(output) == 1:
        def mlp(X):
            return output[0](hmlp(X))
    else:
        def mlp(X):
            h = hmlp(X)
            return [o(h) for o in output]
    return mlp, hidden_params + [out_params]
