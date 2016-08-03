"""
Implements a variational autoencoder to learn a deep generative model 
of MNIST digits, in the style of Kingma + Welling
(https://arxiv.org/abs/1312.6114)
"""
import tensorflow as tf
import numpy as np
from vae import vae, viz, data, nnet

# load data
from tensorflow.examples.tutorials.mnist import input_data
mnist  = input_data.read_data_sets('MNIST_data', one_hot=True)
Xtrain = mnist.train.images
Xtest  = mnist.test.images
Xdim   = Xtrain.shape[1]
viz.plot_random_examples(Xtrain, save=True)

# dimensionality of the latent space
zdim = 2

# create tensorflow session scope
with tf.Session() as sess:

    #################################
    # specify model architecture    #
    #################################
    # create a MLP for the decoder (generative) model (z => X)
    decoder_mlp_layers = [ (zdim, nnet.tanh_layer),
                           (200, nnet.tanh_layer),
                           (200, nnet.sigmoid_layer) ]
    decode, decoder_params = \
        nnet.make_mlp(layers=decoder_mlp_layers, out_dim=Xdim, init_scale=.11)

    # create a MLP for the encoder (recognition/inference model) (X => p(z))
    encoder_layers = [ (Xdim, nnet.tanh_layer),
                       (200, nnet.linear_layer, nnet.linear_layer) ]
    encode, encoder_params = \
        nnet.make_mlp(layers=encoder_layers, out_dim=zdim, init_scale=.11)

    ##########################################
    # make variational lower bound objective #
    ##########################################
    vlb = vae.make_vae_objective(encode, decode, zdim, vae.binary_loglike)

    ######################################################
    # specify validation and test functions for callback #
    ######################################################
    def make_callback():
        Xtest_tf     = tf.constant(Xtest, dtype=tf.float32, shape=Xtest.shape)
        Ntest        = len(Xtest)
        test_vlb     = vlb(Xtest_tf, Ntest, Ntest, 5)
        test_vlb_fun = lambda: test_vlb.eval(session=sess) * Ntest
        def callback(itr):
            print "  test vlb = %2.2f" % np.mean(test_vlb_fun())

    cb = make_callback()

    #########################################################
    # Make inference function - and run with a tf optimizer #
    #########################################################
    fit = vae.make_fitter(vlb, Xtrain, callback=cb, load_data=True)

    ## initialize variables and fit
    sess.run(tf.initialize_all_variables())
    fit(100, 200, 1, tf.train.AdamOptimizer(3e-4), sess)
