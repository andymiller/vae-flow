import tensorflow as tf
import nnet as nn
import util as ut
import numpy as np
from time import time
import pyprind
import numpy.random as npr


def make_fitter(vlb, X, callback=None, load_data=True):
    N, xdim = X.shape

    # load all data onto the gpu at once... ideally
    if load_data:
        X_all = tf.constant(X, name='X')

    def fit(num_epochs, minibatch_size, L, optimizer, sess):
        num_batches = N // minibatch_size

        # set up cost function and updates
        if load_data:
            idx      = tf.placeholder(tf.int32, name='idx')
            mbsize   = tf.constant(minibatch_size)
            xdimsize = tf.constant(xdim)
            x_batch  = tf.slice(X_all, [idx*mbsize, 0],
                                       [mbsize,xdimsize], name='x_batch')
        else:
            x_batch  = tf.placeholder(tf.float32, shape=[minibatch_size, xdim],
                                      name='X')
        cost = -tf.reduce_sum(vlb(x_batch, N, minibatch_size, L))
        train_step = optimizer.minimize(cost)

        sess.run(tf.initialize_variables(ut.nontrainable_variables()))

        def train(bidx):
            if load_data:
                train_step.run(feed_dict={idx:bidx}, session=sess)
                return cost.eval(feed_dict={idx:bidx}, session=sess)
            else:
                xb = X[bidx*minibatch_size:(bidx+1)*minibatch_size]
                train_step.run(feed_dict={x_batch: xb}, session=sess)
                return cost.eval(feed_dict={x_batch: xb}, session=sess)

        start = time()
        for i in xrange(num_epochs):
            bidxs = npr.permutation(num_batches)
            vals = [train(bidx) for bidx in pyprind.prog_bar(bidxs)]
            print 'epoch {:>4} of {:>4}: {:> .6}' . \
                    format(i+1, num_epochs, np.median(vals[-10:]))
            if callback:
                callback(i)

        stop = time()
        print 'cost {}, {:>5} sec per update, {:>5} sec total\n'.format(
            np.median(vals[-10:]), (stop - start) / N, stop - start)

    return fit

#########################
#  objective functions  #
#########################

def binary_loglike(X, p):
    return tf.reduce_sum(tf.log(p)*X + tf.log(1.-p)*(1.-X),
                          reduction_indices=1)
    #var = p * (1 - p)
    #return gaussian_loglike(X, (p, var))


def gaussian_loglike(X, params):
    mu, log_sigmasq = params
    return -0.5*tf.reduce_sum(
            (np.log(2.*np.pi) + log_sigmasq) + (X - mu)**2. / tf.exp(log_sigmasq), 
            reduction_indices=1)


def kl_to_prior(mu, log_sigmasq):
    return -0.5*tf.reduce_sum(1. + log_sigmasq - mu**2. - tf.exp(log_sigmasq), 
                              reduction_indices=1)

def normal_normal_kl(amu, alog_sigmasq, bmu, blog_sigmasq):
    """ compute KL( N_a || N_b ) for two independent gaussian distributions
    adopted from http://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    """
    kls = blog_sigmasq - alog_sigmasq - 1. + \
          tf.exp(alog_sigmasq - blog_sigmasq) + \
          (bmu - amu)**2 * tf.exp(-blog_sigmasq)
    return tf.reduce_sum(.5 * kls, reduction_indices=1)


def make_vae_objective(encode, decode, zdim, loglike):
    def vlb(X, N, M, L):
        """ variational lower bound
            Args:
              - X : Nbatch x dimx data matrix
              - N : total number of data (for overall scale)
              - M : Minibatch Size
              - L : number of samples for monte carlo estimate
        """
        def sample_z(mu, log_sigmasq):
            eps = tf.random_normal((M, zdim), dtype=tf.float32)
            return mu + tf.exp(0.5 * log_sigmasq) * eps

        mu, log_sigmasq = encode(X)
        logpxz = sum(loglike(X, decode(sample_z(mu, log_sigmasq)))
                     for l in xrange(L)) / float(L)

        minibatch_val = -kl_to_prior(mu, log_sigmasq) + logpxz

        return minibatch_val / M  # NOTE: multiply by N for overall vlb
    return vlb


def make_aux_vae_objective(encode, aux_encode, aux_decode, decode, zdim, adim, loglike):
    def vlb(X, N, M, L):
        def sample_normal(mu, log_sigmasq, dim):
            eps = tf.random_normal((M, dim), dtype=tf.float32)
            return mu + tf.exp(0.5 * log_sigmasq) * eps

        # q(a | X) (aux encoder)
        amu, alog_sigmasq = aux_encode(X)
        asamps            = sample_normal(amu, alog_sigmasq, dim=adim)

        # q(z | a, x) (encoder)
        zmu, zlog_sigmasq = encode(tf.concat(1, [X, asamps]))
        zsamps            = sample_normal(zmu, zlog_sigmasq, dim=zdim)

        # p(a | X, z) (aux decoder)
        pamu, palog_sigmasq = aux_decode(tf.concat(1, [X, zsamps]))

        # p(X | z) (decoder)
        logpxz = sum(loglike(X, decode(zsamps)) for l in xrange(L)) / float(L)
        minibatch_val = logpxz \
                        -kl_to_prior(zmu, zlog_sigmasq) \
                        -normal_normal_kl(amu, alog_sigmasq, pamu, palog_sigmasq)

        return minibatch_val / M
    return vlb
