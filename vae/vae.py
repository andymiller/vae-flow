import tensorflow as tf
import nnet as nn
import util as ut
from time import time
import numpy as np

def init_binary_objective(X, zdim, encoder_hdims, decoder_hdims, callback=None):
    """ initializes VAE model that maps from a gaussian latent space to a
    binary vector whose values are all in [0, 1] """
    assert X.ndim==2
    N, xdim = X.shape

    # initialize model params
    encoder_params, decoder_params, all_params = \
        init_binary_params(xdim, zdim, encoder_hdims, decoder_hdims)

    # create the (monte carlo) evidence lower bound function
    vlb = make_binary_objective(encoder_params, decoder_params)

    # create a function that optimizes VLB
    #fit = make_fitter(vlb, (encoder_params, decoder_params), X)
    return encoder_params, decoder_params, vlb # fit


def make_fitter(vlb, params, X, callback=None):
    N, xdim = X.shape

    def fit(num_epochs, minibatch_size, L, optimizer, sess):
        num_batches = N // minibatch_size

        # set up cost function and updates
        x_batch    = tf.placeholder(tf.float32, shape=[minibatch_size, xdim], name='X')
        cost       = -tf.reduce_sum(vlb(x_batch, N, minibatch_size, L))
        train_step = optimizer.minimize(cost)

        def train(idx):
            xb = X[idx*minibatch_size:(idx+1)*minibatch_size]
            train_step.run(feed_dict={x_batch: xb})
            return cost.eval(feed_dict={x_batch: xb})

        sess.run(tf.initialize_all_variables())
        start = time()
        for i in xrange(num_epochs):
            vals = [train(bidx) for bidx in xrange(num_batches)]
            print 'epoch {:>4} of {:>4}: {:> .6}'.format(i+1, num_epochs, np.median(vals[-10:]))

        stop = time()
        print 'cost {}, {} sec per update, {} sec total\n'.format(
            np.median(vals[-10:]), (stop - start) / N, stop - start)

    return fit


############################
# parameter initialization #
############################


def init_mlp_params(N_in, hdims, N_out, gaussian_output=True):
    """initializes MLP parameters from N_in => N_out with hdims specified, 
    where the output is real valued (gaussian_output=True) or squashed"""
    dims = [N_in] + hdims
    nnet_params = [nn.init_layer(shape, layer_name="h_%d"%i)
                   for i,shape in enumerate(zip(dims[:-1], dims[1:]))]
    if gaussian_output:
        W_mu, b_mu       = nn.init_layer((hdims[-1], N_out), layer_name="out_mu")
        W_sigma, b_sigma = nn.init_layer((hdims[-1], N_out), layer_name="out_sig")
        out = [(W_mu, b_mu), (W_sigma, b_sigma)]
    else:
        out = [nn.init_layer((hdims[-1], N_out), layer_name="out")]
    return nnet_params + out 

def _make_initializer(init_decoder):
    def init_params(Nx, Nz, encoder_hdims, decoder_hdims):
        encoder_params = init_mlp_params(Nx, encoder_hdims, Nz, gaussian_output=True)
        decoder_params = init_decoder(Nz, decoder_hdims, Nx)
        return encoder_params, decoder_params, None #, \
               #ut.flatten((encoder_params, decoder_params))
    return init_params

init_gaussian_params = _make_initializer(init_mlp_params)
init_binary_params   = _make_initializer(
    lambda nin,h,nout: init_mlp_params(nin, h, nout, gaussian_output=False))


##########################
#  enoders and decoders  #
##########################


def unpack_gaussian_params(mlp_params):
    nnet_params, ((W_mu, b_mu), (W_sigma, b_sigma)) = \
        mlp_params[:-2], mlp_params[-2:]
    return nnet_params, (W_mu, b_mu), (W_sigma, b_sigma)


def unpack_binary_params(mlp_params):
    nnet_params, (W_out, b_out) = mlp_params[:-1], mlp_params[-1]
    return nnet_params, (W_out, b_out)


def encoder(encoder_params):
    'a neural net with tanh layers until the final layer,'
    'which generates mu and log_sigmasq separately'
    nnet_params, (W_mu, b_mu), (W_sigma, b_sigma) = \
        unpack_gaussian_params(encoder_params)

    nnet        = nn.compose(nn.tanh_layer(W, b) for W, b in nnet_params)
    mu          = nn.linear_layer(W_mu, b_mu)
    log_sigmasq = nn.linear_layer(W_sigma, b_sigma)

    def encode(X):
        h = nnet(X)
        return mu(h), log_sigmasq(h)

    return encode


def gaussian_decoder(decoder_params):
    'just like the (gaussian) encoder but means are mapped through a logistic'

    code = encoder(decoder_params)

    def decode(Z):
        mu, log_sigmasq = code(Z)
        return tf.nn.sigmoid(mu), log_sigmasq

    return decode


def binary_decoder(decoder_params):
    'a neural net with tanh layers until the final sigmoid layer'

    nnet_params, (W_out, b_out) = unpack_binary_params(decoder_params)

    nnet = nn.compose(nn.tanh_layer(W, b) for W, b in nnet_params)
    Y    = nn.sigmoid_layer(W_out, b_out)

    def decode(Z):
        return Y(nnet(Z))

    return decode


#########################
#  objective functions  #
#########################

def binary_loglike(X, Y):
    return -tf.reduce_sum( (X-Y)**2., reduction_indices=1)


def gaussian_loglike(X, params):
    mu, log_sigmasq = params
    return -0.5*tf.reduce_sum(
            (np.log(2.*np.pi) + log_sigmasq) + (X - mu)**2. / tf.exp(log_sigmasq), 
            reduction_indices=1)


def kl_to_prior(mu, log_sigmasq):
    return -0.5*tf.reduce_sum(1. + log_sigmasq - mu**2. - tf.exp(log_sigmasq), 
                              reduction_indices=1)


def _make_objective(decoder, loglike):
    def make_objective(encoder_params, decoder_params):
        encode = encoder(encoder_params)
        decode = decoder(decoder_params)
        z_dim = get_zdim(decoder_params)

        def vlb(X, N, M, L):
            """ variational lower bound
                Args:
                  - X : Nbatch x dimx data matrix
                  - N : total number of data (for overall scale)
                  - M : Minibatch Size
                  - L : number of samples for monte carlo estimate
            """
            def sample_z(mu, log_sigmasq):
                eps = tf.random_normal((M, z_dim), dtype=tf.float32)
                return mu + tf.exp(0.5 * log_sigmasq) * eps

            mu, log_sigmasq = encode(X)
            logpxz = sum(loglike(X, decode(sample_z(mu, log_sigmasq)))
                         for l in xrange(L)) / float(L)

            minibatch_val = -kl_to_prior(mu, log_sigmasq) + logpxz

            return minibatch_val / M  # NOTE: multiply by N for overall vlb
        return vlb
    return make_objective

make_gaussian_objective = _make_objective(gaussian_decoder, gaussian_loglike)
make_binary_objective   = _make_objective(binary_decoder, binary_loglike)


##########
#  util  #
##########

def get_zdim(decoder_params):
    try:
        return decoder_params[0][0].get_shape()[0].value
    except AttributeError:
        return decoder_params[0][0].shape[0]


