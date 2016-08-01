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
    def make_sample_fun():
        nsamps = tf.placeholder(tf.int32, name='nsamps')
        sfun = decode(tf.random_normal((nsamps, zdim), dtype=tf.float32))
        def sample_fun(num_samps):
            return sfun.eval(feed_dict={nsamps:num_samps}, session=sess)
        return sample_fun
    sample_fun = make_sample_fun()

    # reconstruction function
    def make_recon_fun():
        Xsub       = tf.placeholder(tf.float32, shape=[1, Xdim], name='Xsub')
        nsamps = tf.placeholder(tf.int32, name='nsamps')
        zm, zls    = encode(Xsub)
        recon_imgs = decode(vae.sample_normal(zm, zls, nsamps=nsamps-1))
        def recon_fun(num_samps):
            # choose one datapoint, encode it
            subset = X[np.random.choice(X.shape[0], 1)]
            imgs   = recon_imgs.eval(session=sess, feed_dict={Xsub: subset, nsamps:num_samps})
            return np.row_stack([subset, imgs])
        return recon_fun
    recon_fun = make_recon_fun()

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
        fit(100, 200, 1, tf.train.AdamOptimizer(3e-4), sess)

        # save model
        saver = tf.train.Saver()
        saver.save(sess, "mnist_vae_z_%d.ckpt" % (zdim))

        #########################################
        # encode train data into latent space   #
        #########################################
        Xtrain = tf.placeholder(tf.float32, shape=X.shape, name='Xin')
        zmu_train, zls_train = encode(Xtrain)
        ztrain = vae.sample_normal(zmu_train, zls_train, nsamps=1)
        zt = ztrain.eval(session=sess, feed_dict={Xtrain:X})

        # fit a mixture model to zt, and sample from GMM prior
        from sklearn.mixture import GMM
        gmm = GMM(n_components=20, covariance_type='full')
        gmm.fit(zt)

        def make_gmm_sample_fun():
            z_gmm    = tf.placeholder(tf.float32, shape=(None, zdim))
            sfun_gmm = decode(z_gmm)
            def gmm_sample_fun(num_samps):
                return sfun_gmm.eval(feed_dict={z_gmm: gmm.sample(num_samps)}, session=sess)
            return gmm_sample_fun
        gmm_sample_fun = make_gmm_sample_fun()

        viz.plot_samples(0, gmm_sample_fun, savedir=OUTPUT_DIR, stub='gmm_', sidelen=10)
        viz.plot_samples(0, sample_fun, savedir=OUTPUT_DIR, stub='big_', sidelen=10)

        import matplotlib.pyplot as plt
        xg = yg = np.linspace(-3, 3, 100)
        xx, yy = np.meshgrid(xg, yg)
        fig = plt.figure(figsize=(8,8))
        z = gmm.score(np.column_stack([xx.flatten(), yy.flatten()]))
        plt.contourf(xx, yy, np.exp(z.reshape(xx.shape)))
        import os
        plt.savefig(os.path.join(OUTPUT_DIR, 'prob_z_gmm.png'))
        plt.close("all")



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




