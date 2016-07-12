import tensorflow as tf
import numpy as np
from vae import vae, viz, data, nnet

# load binarized mnist
from tensorflow.examples.tutorials.mnist import input_data
bmnist = data.binarized_mnist()
X      = bmnist[0]
Xt     = bmnist[2]
viz.plot_random_examples(X, save=True)

# initialize session and vars
sess = tf.InteractiveSession()


##############################################################################
# Forward and recognition models
#
#    Each layer in the MLP is specified (hidden-dim, layer-type)
#    so a [(10, tanh), (20, tanh)] with outputsize 2
#    implies a set of parameters
#       W_1 (10 x 20), b_1 (20 x 1)
#       W_2 (20 x 2),  b_2 (2 x 1)
#
##############################################################################
zdim = 10
Xdim = X.shape[1]
decoder_mlp_layers = [ (zdim, nnet.relu_layer),
                       (200, nnet.stochastic_tanh_layer),
                       (200, nnet.sigmoid_layer) ]
decode, decoder_params = \
    nnet.make_mlp(layers = decoder_mlp_layers, out_dim = Xdim)

encoder_mlp_layers = [ (Xdim, nnet.tanh_layer),
                       #(200, nnet.tanh_layer),
                       (200, nnet.linear_layer, nnet.linear_layer) ]
encode, encoder_params = \
    nnet.make_mlp(layers = encoder_mlp_layers, out_dim = zdim)

#######################################
# make VLB for binary objective       #
#######################################
vlb = vae.make_vae_objective(encode, decode, zdim, vae.binary_loglike)

###################################
# validation dataset VLB          #
###################################
Xtest = tf.constant(Xt)
Ntest = len(Xt)
test_lb_fun = vlb(Xtest, Ntest, Ntest, 5)

#################################
# training callback function    #
#################################
def callback(itr):
    def samplefun(num_samps):
        import numpy as np
        z = np.array(np.random.randn(num_samps, zdim), dtype=np.float32)
        return decode(z).eval(session=sess)
    viz.plot_samples(itr, samplefun, savedir='vae_mnist_samples')

    def sample_z(mu, log_sigmasq, M=5):
        eps = tf.random_normal((M, zdim), dtype=tf.float32)
        return mu + tf.exp(0.5 * log_sigmasq) * eps

    def recons(num_samps):
        # random subset
        subset = X[np.random.choice(X.shape[0], 1)]
        mu, log_sigmasq = encode(subset)
        imgs = decode(sample_z(mu, log_sigmasq, M=24)).eval(session=sess)
        return np.row_stack([subset, imgs])
    viz.plot_samples(itr, recons, savedir='vae_mnist_samples', stub='recon')
    test_lb = test_lb_fun.eval(session=sess) * Ntest
    print "test data VLB: ", np.mean(test_lb)

##########################################
# Make gradient descent fitting function #
##########################################
fit = vae.make_fitter(vlb, X, callback, load_data=False)

## initialize variables and fit
sess.run(tf.initialize_all_variables())
#fit(10, 10, 1, tf.train.AdamOptimizer(1e-3), sess)
#fit(10, 50, 1, tf.train.GradientDescentOptimizer(1e-2), sess)
fit(100, 50, 5, tf.train.AdamOptimizer(1e-4), sess)
fit(100, 100, 1, tf.train.AdamOptimizer(1e-4), sess)

#fit(100, 100, 1, tf.train.AdamOptimizer(1e-4), sess)

#
