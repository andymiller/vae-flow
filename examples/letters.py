import tensorflow as tf
from vae import vae

# load TF data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
X = mnist.test.images

#from vae import init_mlp_params
#bin_p = init_mlp_params(100, [20, 20], 2, gaussian_output=

# create encoder/decoder/fitter
encoder_params, decoder_params, fit, vlb = \
    vae.make_binary_fitter(X    = X,
                           zdim = 2,
                           encoder_hdims = [200],
                           decoder_hdims = [200])

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

a = vlb(X[:10], 1, 10, 10)
