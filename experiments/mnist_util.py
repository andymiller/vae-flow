from vae import viz
import numpy as np


def load_binarized_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    np.random.seed(42)
    binarize = lambda x: np.array(np.random.rand(*x.shape) < x, dtype=np.float32)
    X, Y     = binarize(mnist.train.images),  mnist.train.labels
    Xt, Yt   = binarize(mnist.test.images), mnist.test.labels
    return (X, Y), (Xt, Yt)


def make_callback(sample_fun, recon_fun, latent_space_fun,
                  test_vlb_fun, output_dir):
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
    import seaborn as sns;
    sns.set_style("white")
    import os
    colors = sns.color_palette('Set3', n_colors=10)
    def callback(itr):

        # plot samples from the model generative process
        viz.plot_samples(itr, sample_fun, savedir=output_dir)

        # plot reconstructed samples
        viz.plot_samples(itr, recon_fun, savedir=output_dir, stub='recon')

        # evaluate test 
        test_lb = test_vlb_fun()
        print "test data VLB: ", np.mean(test_lb)

        # plot z space for each test data item, colored by digit type
        def plot_test_latent():
            Z, Y = latent_space_fun()
            fig  = plt.figure()
            ax   = plt.gca()
            for i in xrange(10):
                idx = np.argmax(Y, 1) == i
                ax.scatter(Z[idx,0], Z[idx,1], c=colors[i], label='%d'%i)
            ax.legend()
            fig.savefig(os.path.join(output_dir, "latent_space_%03d.png" % itr))
        plot_test_latent()
    return callback


