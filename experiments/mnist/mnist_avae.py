"""
Implementation of the auxiliary VAE: https://arxiv.org/pdf/1602.05473v4.pdf

    The aux vae defines a joint model over latent variables (z),
    aux variables (a), and data (X)

      p(a, z, X) = p(X | z) p(a | X, z) p(z)

    and uses an inference network in an SVAE setting with the form

      q(a, z | X) = q(a | X) q(z | a, X)

    The main idea here is that the auxiliary variables don't contribute
    to the data generating process, but do help the inference network
    use a more flexible posterior approximation (essentially by being a
    hierarchical model-as-approximation)

"""
import tensorflow as tf
import numpy as np
from vae import vae, viz, data, nnet

def load_binarized_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    np.random.seed(42)
    binarize = lambda x: np.array(np.random.rand(*x.shape) < x, dtype=np.float32)
    X, Y     = binarize(mnist.train.images),  mnist.train.labels
    Xt, Yt   = binarize(mnist.test.images), mnist.test.labels
    return (X, Y), (Xt, Yt)

def make_eval_callback(sample_fun, recon_fun, latent_space_fun,
                       test_vlb_fun, output_dir, sess):
    """ create an evaluation callback function for training --- functions
    passed in are:
        sample_fun(num_samps):   returns num_samps x 784 array of samples
                                 from prior
        recon_fun(x, num_samps): returns num_samps x 784 array of
                                 reconstructions from posterior over x
                                 (single datum)
        held_out_latent_space  : returns held out latent space z values

    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    colors = sns.color_palette('muted', n_colors=10)
    def callback(itr):

        # plot samples from the model generative process
        viz.plot_samples(itr, sample_fun, savedir=output_dir)

        # plot reconstructed samples
        viz.plot_samples(itr, recon_fun, savedir=output_dir, stub='recon')

        # evaluate test 
        test_lb = test_vlb_fun.eval(session=sess) * Ntest
        print "test data VLB: ", np.mean(test_lb)

        # plot z space for each test data item, colored by digit type
        def plot_test_latent():
            z = latent_space_fun()
            fig = plt.figure()
            ax  = plt.gca()
            for i in xrange(10):
                idx = np.argmax(Yt, 1) == i
                ax.scatter(z[idx,0], z[idx,1], c=colors[i], label='%d'%i)
            ax.legend()
            fig.savefig(os.path.join(output_dir, "latent_space_%03d.png" % itr))
        plot_test_latent()
    return callback


######################################################################
# Script params - if EVAL_MODEL is true, bypass training, load in
# tensorflow checkpointed model, run evaluation blocks
######################################################################
EVAL_MODEL = False
zdim = 2
adim = 10


if __name__=="__main__":

    # data
    (X, Y), (Xt, Yt) = load_binarized_mnist()
    viz.plot_random_examples(X, save=True)

    # initialize session and vars
    sess = tf.InteractiveSession()

    ##################################################################
    # forward, aux-forward, aux-recognition and z recognition models #
    ##################################################################
    Xdim = X.shape[1]

    # z => X
    decoder_mlp_layers = [ (zdim, nnet.tanh_layer),
                           (200, nnet.tanh_layer),
                           (200, nnet.sigmoid_layer) ]
    decode, decoder_params = \
        nnet.make_mlp(layers = decoder_mlp_layers, out_dim = Xdim)

    # X, z => a
    aux_decoder_layers = [ (Xdim + zdim, nnet.tanh_layer),
                           (200, nnet.linear_layer, nnet.linear_layer) ]
    aux_decode, aux_decoder_params = \
        nnet.make_mlp(layers=aux_decoder_layers, out_dim = adim)

    # X => a
    aux_encoder_layers = [ (Xdim, nnet.tanh_layer),
                           (200, nnet.linear_layer, nnet.linear_layer) ]
    aux_encode, aux_encoder_params = \
        nnet.make_mlp(layers = aux_encoder_layers, out_dim = adim)

    # a, X => z
    encoder_layers = [ (Xdim + adim, nnet.tanh_layer),
                       (200, nnet.tanh_layer),
                       (200, nnet.linear_layer, nnet.linear_layer) ]
    encode, encoder_params = \
        nnet.make_mlp(layers = encoder_layers, out_dim = zdim)

    #######################################
    # make VLB for binary objective       #
    #######################################
    vlb = vae.make_aux_vae_objective(encode, aux_encode, aux_decode, decode,
                                     zdim, adim, vae.binary_loglike)

    ###################################
    # validation dataset VLB          #
    ###################################
    Xtest = tf.constant(Xt)
    Ntest = len(Xt)
    test_lb_fun = vlb(Xtest, Ntest, Ntest, 5)

    ########################################################################
    # training callback function - does essentially all model evaluation   #
    ########################################################################
    def callback(itr):
        def samplefun(num_samps):
            import numpy as np
            z = np.array(np.random.randn(num_samps, zdim), dtype=np.float32)
            return decode(z).eval(session=sess)
        viz.plot_samples(itr, samplefun, savedir='avae_mnist_samples')

        def sample_normal(mu, log_sigmasq, M=5):
            dim = mu.get_shape()[1].value
            eps = tf.random_normal((M, dim), dtype=tf.float32)
            return mu + tf.exp(0.5 * log_sigmasq) * eps

        def recons(num_samps):
            # choose one datapoint, encode it
            subset = X[np.random.choice(X.shape[0], 1)]
            # compute encoder - first aux, then encoder conditioned on aux
            amu, alog_sigmasq = aux_encode(subset)
            asamps            = sample_normal(amu, alog_sigmasq, M=24)
            Xa                = tf.concat(1, [tf.tile(subset, [24, 1]),  asamps])
            zmu, zlog_sigmasq = encode(Xa)
            imgs = decode(sample_normal(zmu, zlog_sigmasq, M=24)).eval(session=sess)
            return np.row_stack([subset, imgs])

        viz.plot_samples(itr, recons, savedir='avae_mnist_samples', stub='recon')
        test_lb = test_lb_fun.eval(session=sess) * Ntest
        print "test data VLB: ", np.mean(test_lb)

        # plot z space for each test data item, colored by digit type
        def plot_test_latent():
            amu, alog_sigmasq = aux_encode(Xtest)
            asamps            = sample_normal(amu, alog_sigmasq, M=1)
            zmu, zlog_sigmasq = encode(tf.concat(1, [Xtest, asamps]))
            z                 = sample_normal(zmu, zlog_sigmasq, M=1).eval(session=sess)
            print z.shape
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax  = plt.gca()
            import seaborn as sns
            colors = sns.color_palette('muted', n_colors=10)
            for i in xrange(10):
                idx = np.argmax(Yt, 1) == i
                ax.scatter(z[idx,0], z[idx,1], c=colors[i], label='%d'%i)
            ax.legend()
            fig.savefig("avae_mnist_samples/latent_space_%03d.png" % itr)

        plot_test_latent()


    ##########################################
    # Make gradient descent fitting function #
    ##########################################
    fit = vae.make_fitter(vlb, X, callback, load_data=False)
    if not EVAL_MODEL:
        ## initialize variables and fit
        sess.run(tf.initialize_all_variables())
        #fit(10, 50, 1, tf.train.GradientDescentOptimizer(1e-2), sess)
        #fit(10, 10, 1, tf.train.AdamOptimizer(1e-3), sess)
        fit(1000, 200, 1, tf.train.AdamOptimizer(3e-4), sess)

    else:
        saver = tf.train.Saver()
        saver.restore(sess, "mnist_avae_z_%d_a_%d.ckpt" % (zdim, adim))

        # plot the latent space labeled
        def sample_normal(mu, log_sigmasq, M=5):
            dim = mu.get_shape()[1].value
            eps = tf.random_normal((M, dim), dtype=tf.float32)
            return mu + tf.exp(0.5 * log_sigmasq) * eps

        amu, alog_sigmasq = aux_encode(X)
        asamps            = sample_normal(amu, alog_sigmasq, M=1)
        zmu, zlog_sigmasq = encode(tf.concat(1, [X,  asamps]))
        zmu = zmu.eval(session=sess)

        import matplotlib.pyplot as plt; plt.ion()
        import seaborn as sns
        plt.scatter(z[:,0], z[:,1])

        # post expeirment evaluation
        # https://github.com/yburda/iwae/blob/master/experiments.py#L42
        #  project 

        # project z's to two dimension
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        z = pca.fit_transform(zmu)




