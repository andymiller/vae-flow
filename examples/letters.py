import tensorflow as tf
from vae import vae

# load TF data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
X = mnist.test.images

# initialize session and vars
sess = tf.InteractiveSession()

# create encoder, decoder, and variational lower bound
encoder_params, decoder_params, vlb = \
    vae.init_binary_objective(X    = X,
			      zdim = 2,
			      encoder_hdims = [200],
			      decoder_hdims = [200])

fit = vae.make_fitter(vlb, (encoder_params, decoder_params), X[:2000])

# TODO: figure out how to initialize variables before training - make it not depend on session somehow....
# initialize session and evaluate variational lower bound
sess.run(tf.initialize_all_variables())

fit(10, 50, 10, tf.train.AdamOptimizer(1e-4), sess)


