import tensorflow as tf
import numpy as np
from vae import vae, viz

# load mnist letters
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
X     = mnist.train.images
#Xtest = mnist.test.images[:3000,:]
viz.plot_random_examples(X, save=True)

# initialize session and vars
sess = tf.Session()

# create encoder, decoder, and variational lower bound
zdim = 10
encode, decode, vlb = \
    vae.init_binary_objective(X    = X,
                              zdim = zdim,
                              encoder_hdims = [300],
                              decoder_hdims = [300])

Xtest = tf.constant(mnist.test.images)
Ntest = len(mnist.test.images)
test_lb_fun = vlb(Xtest, Ntest, Ntest, 5)

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

fit = vae.make_fitter(vlb, X, callback, load_data=False)

## initialize variables and fit
sess.run(tf.initialize_all_variables())
#fit(10, 50, 2, tf.train.AdamOptimizer(1e-3), sess)
fit(200, 50, 2, tf.train.GradientDescentOptimizer(1e-3), sess)

