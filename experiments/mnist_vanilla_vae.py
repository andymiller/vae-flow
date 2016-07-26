"""
Vanilla variational autoencoder experiments on MNIST
"""
import tensorflow as tf
import numpy as np
from vae import vae, viz, data, nnet
from mnist_util import load_binarized_mnist, make_callback

######################################################################
# Script params - if EVAL_MODEL is true, bypass training, load in
# tensorflow checkpointed model, run evaluation blocks
######################################################################
EVAL_MODEL = False
OUTPUT_DIR = 'vae_mnist_samples'
zdim = 2

if __name__=="__main__":

    # data
    (X, Y), (Xt, Yt) = load_binarized_mnist()
    viz.plot_random_examples(X, save=True)

    # initialize session and vars
    sess = tf.Session()

    ##################################################################
    # forward, aux-forward, aux-recognition and z recognition models #
    ##################################################################
    Xdim = X.shape[1]

    # z => X
    decoder_mlp_layers = [ (zdim, nnet.tanh_layer),
                           (200, nnet.tanh_layer),
                           (200, nnet.sigmoid_layer) ]
    decode, decoder_params = \
        nnet.make_mlp(layers = decoder_mlp_layers, out_dim = Xdim, init_scale=.11)

    # X => a
    encoder_layers = [ (Xdim, nnet.tanh_layer),
                       (200, nnet.linear_layer, nnet.linear_layer) ]
    encode, encoder_params = \
        nnet.make_mlp(layers = encoder_layers, out_dim = zdim, init_scale=.11)

    #######################################
    # make VLB for binary objective       #
    #######################################
    vlb = vae.make_vae_objective(encode, decode, zdim, vae.binary_loglike)

    ###################################
    # validation and test functions   #
    ###################################

    # test data bound
    Xtest = tf.constant(Xt, dtype=tf.float32, shape=Xt.shape, name='Xtest')
    Ntest = len(Xt)
    test_vlb     = vlb(Xtest, Ntest, Ntest, 5)
    test_vlb_fun = lambda: test_vlb.eval(session=sess) * Ntest

    # sample from prior function
    nsamps = tf.placeholder(tf.int32, name='nsamps')
    sfun = decode(tf.random_normal((nsamps, zdim), dtype=tf.float32))
    def sample_fun(num_samps):
        return sfun.eval(feed_dict={nsamps:num_samps}, session=sess)

    # reconstruction function
    Xsub       = tf.placeholder(tf.float32, shape=[1, Xdim], name='Xsub')
    zm, zls    = encode(Xsub)
    recon_imgs = decode(vae.sample_normal(zm, zls, nsamps=nsamps-1))
    def recon_fun(num_samps):
        # choose one datapoint, encode it
        subset = X[np.random.choice(X.shape[0], 1)]
        imgs   = recon_imgs.eval(session=sess, feed_dict={Xsub: subset, nsamps:num_samps})
        return np.row_stack([subset, imgs])

    # encode test data into latent space
    zmu_test, zls_test = encode(Xtest)
    ztest = vae.sample_normal(zmu_test, zls_test, nsamps=1)
    def latent_space_fun():
        return ztest.eval(session=sess), Yt

    # training callback function - does essentially all model evaluation
    callback = make_callback(sample_fun       = sample_fun,
                             recon_fun        = recon_fun,
                             latent_space_fun = latent_space_fun,
                             test_vlb_fun     = test_vlb_fun,
                             output_dir       = OUTPUT_DIR)

    ##########################################
    # Make gradient descent fitting function #
    ##########################################
    fit = vae.make_fitter(vlb, X, callback, load_data=True)
    if not EVAL_MODEL:
        ## initialize variables and fit
        sess.run(tf.initialize_all_variables())
        fit(1000, 200, 1, tf.train.AdamOptimizer(3e-4), sess)

    else:
        saver = tf.train.Saver()
        saver.restore(sess, "mnist_avae_z_%d_a_%d.ckpt" % (zdim, adim))

        # plot the latent space labeled
        amu, alog_sigmasq = aux_encode(X)
        asamps            = vae.sample_normal(amu, alog_sigmasq, M=1)
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




