"""
Implements a variational autoencoder to learn a deep generative model 
of MNIST digits, in the style of Kingma + Welling
(https://arxiv.org/abs/1312.6114)
"""
import tensorflow as tf
import numpy as np
from vae import vae, viz, data, nnet
from tensorflow.contrib import slim

# load data
from tensorflow.examples.tutorials.mnist import input_data
mnist  = input_data.read_data_sets('MNIST_data', one_hot=True)
Xtrain = mnist.train.images
Xtest  = mnist.test.images
Xdim   = Xtrain.shape[1]
viz.plot_random_examples(Xtrain, save=True)

N_MINIBATCH = 128

#################################################################
# generative and inference networks, using tensorflow slim      #
#################################################################


def generative_network(z, zdim):
  """Generative network to parameterize generative model. It takes
  latent variables as input and outputs the likelihood parameters.
  logits = neural_network(z)

  Args:
  z = tensor input
  d = latent variable dimension
  """
  with slim.arg_scope([slim.conv2d_transpose],
                      activation_fn=tf.nn.elu,
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={'scale': True}):
    net = tf.reshape(z, [N_MINIBATCH, 1, 1, zdim])
    net = slim.conv2d_transpose(net, 128, 3, padding='VALID')
    net = slim.conv2d_transpose(net, 64, 5, padding='VALID')
    net = slim.conv2d_transpose(net, 32, 5, stride=2)
    net = slim.conv2d_transpose(net, 1, 5, stride=2, activation_fn=None)
    net = slim.flatten(net)
    #net = slim.nn.sigmoid(net)
    return net


def inference_network(x, xwidth=28, xheight=28, zdim=2):
  """Inference network to parameterize variational model. It takes
  data as input and outputs the variational parameters.
  mu, sigma = neural_network(x)
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.elu,
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={'scale': True}):
    net = tf.reshape(x, [N_MINIBATCH, 28, 28, 1])
    net = slim.conv2d(net, 32, 5, stride=2)
    net = slim.conv2d(net, 64, 5, stride=2)
    net = slim.conv2d(net, 128, 5, padding='VALID')
    net = slim.dropout(net, 0.9)
    net = slim.flatten(net)
    params = slim.fully_connected(net, zdim * 2, activation_fn=None)

  mu    = params[:, :zdim]
  #sigma = tf.nn.softplus(params[:, zdim:])
  sigma = params[:, zdim:]
  return mu, sigma


##########################################
# make variational lower bound objective #
##########################################
zdim   = 2
decode = lambda z: generative_network(z, zdim)
encode = lambda x: inference_network(x)

def gaussian_ll(X, mu):
    return -.5*tf.reduce_sum((np.log(2.*np.pi)) + (X - mu)**2., reduction_indices=1)

def bin_loglike(X, logits):
    return -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits, X),
                          reduction_indices=1)

vlb = vae.make_vae_objective(encode, decode, zdim, bin_loglike) #vae.binary_loglike)

# functions for the lower bound
xin = tf.placeholder(tf.float32, shape=[N_MINIBATCH, 784])
lower_bound = vlb(xin, 2)

xtest0, _ = mnist.test.next_batch(N_MINIBATCH)
xtest0 = np.clip(xtest0, 1e-6, 1-1e-6)
test_vlb = lambda: lower_bound.eval(feed_dict={xin: xtest0})
def callback(itr):
    print "test vlb = %2.2f" % np.mean(test_vlb())

##########################################################
## Make inference function - and run with a tf optimizer #
##########################################################
fit = vae.make_fitter(vlb, Xtrain, callback=callback, load_data=False)

lower_bound = vlb(xin, 10)

zin = tf.placeholder(tf.float32, shape=[N_MINIBATCH, zdim])
Xout = generative_network(zin, 2)

zmu_out, zsigma_out  = inference_network(xin)


## vlb cost
L          = 2
N          = Xtrain.shape[0]
cost       = -tf.reduce_mean(lower_bound) * N
optimizer  = tf.train.AdamOptimizer(1e-2)
train_step = optimizer.minimize(cost)


# create tensorflow session scope
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for i in xrange(10):

        xbatch, _ = mnist.train.next_batch(N_MINIBATCH)
        xbatch = np.clip(xbatch, 1e-6, 1-1e-6)
        batch_cost, train_out = sess.run([cost, train_step],
                                         feed_dict={xin: xbatch})
        print -batch_cost/N, train_out

        # do a recog
        zmu_conc, zsig_conc = sess.run([zmu_out, zsigma_out],
                                        feed_dict={xin: xbatch})
        print zmu_conc.min(), zmu_conc.max(), np.isnan(zmu_conc).sum()
        print zsig_conc.min(), zsig_conc.max(), np.isnan(zsig_conc).sum()

        # do a gen
        Xout_conc = Xout.eval(feed_dict={zin:zmu_conc})
        print Xout_conc.min(), Xout_conc.max(), np.isnan(Xout_conc).sum()

