"""
Implementation of a one-to-many RNN as a generative model for sequences.  

  Z ~ Normal(0, I)
  h0 =

"""
import tensorflow as tf
import numpy as np
from tensorflow import nn
from vae.models import lstm_vae

def gen_dataset(n_data, max_length):
    def gen_seq(w, T):
        tgrid = np.linspace(0, 10, max_length)
        temp1 = np.column_stack([ tgrid * np.sin(tgrid),
                                  tgrid * np.cos(tgrid) ])
        temp2 = np.column_stack([ np.sin(tgrid), np.cos(tgrid) ])
        temp3 = np.column_stack([ np.sin(tgrid)**2, np.cos(tgrid)**2 ])

        seq = w[0]*temp1 + w[1]*temp2 + w[2]*temp3
        return seq[-T:]

    Ws       = .1 * np.random.randn(n_data, 3)
    seq_lens = np.random.binomial(max_length, .7, Ws.shape[0])
    seqs = [ gen_seq(w, T) for w, T in zip(Ws, seq_lens) ]
    return seqs

def pad_seqs(seqs):
    max_length  = np.max([len(s) for s in seqs])
    seqs_padded = np.zeros((len(seqs), max_length, seqs[0].shape[1]))
    for i, s in enumerate(seqs):
        seqs_padded[i,:len(s),:] = s
    return seqs_padded, np.array([len(s) for s in seqs])

if __name__=="__main__":

    ## set up data parameters
    np.random.seed(42)
    num_hidden = 10

    ## generate dataset
    datas = gen_dataset(n_data=100, max_length=100)
    datas_padded, datas_lengths = pad_seqs(datas)
    N, max_length, data_dim = datas_padded.shape

    # create generative functions
    decode = lstm_vae.make_lstm_decoder(num_hidden, data_dim, max_length)
    encode = lstm_vae.make_lstm_encoder(num_hidden, data_dim,
                                        encoder_hidden=10, max_length=max_length)
    vlb    = lstm_vae.make_vae_objective(encode, decode, num_hidden)

    # set up data
    seqs        = tf.placeholder(tf.float32, shape=[None, max_length, data_dim])
    seq_lengths = tf.placeholder(tf.int32, shape=[None])

    # set up cost function
    cost       = -1 * tf.reduce_mean(vlb(seqs, 4, seq_lengths), 0)
    train_step = tf.train.AdamOptimizer(.01).minimize(cost)

    # set up encoder for looking at the model output
    with tf.variable_scope("") as scope:
        scope.reuse_variables()
        zmu, zlogsigmasq = encode(seqs, seq_lengths)

    # set up data generating function in python
    with tf.variable_scope("") as scope:
        scope.reuse_variables()
        latent_vars = tf.placeholder(tf.float32, [None, num_hidden])
        decode_out  = decode(latent_vars)

    ### set up sess and initalize vars ###
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)

    for v in tf.all_variables():
        print v.name

    #############################################
    # set up python wrapping functions          #
    #############################################
    def compute_posterior(datas, datas_len):
        mu, lnsig2 = sess.run([zmu, zlogsigmasq],
                              feed_dict={seqs:datas, seq_lengths:datas_len})
        return mu, lnsig2

    def gen_sample(n_samps, seq_len, eps=None):
        if eps is None:
            eps = np.random.randn(n_samps, num_hidden)
        out = decode_out.eval(feed_dict={latent_vars: eps})
        return out[:,:seq_len,:]

    def posterior_predictive(n_samps, data, data_len):
        mu, lnsig2 = compute_posterior(np.array([data]), np.array([data_len]))
        z = mu + np.exp(.5*lnsig2) * np.random.randn(n_samps, num_hidden)
        out = decode_out.eval(feed_dict={latent_vars: z})
        return out[:,:data_len,:]

    ## run cost on a singel batch
    #ll = cost.eval(feed_dict={seqs: datas_padded, seq_lengths: datas_lengths})

    ######################################
    # optimize and plot reconstructions  #
    ######################################
    import matplotlib
    import matplotlib.pyplot as plt; plt.ion()
    fig, axarr = plt.subplots(2, 3, figsize=(15, 6))

    def plot_data(idx, axarr):
        # plot datas 0
        samps = posterior_predictive(5, datas_padded[idx], datas_lengths[idx])
        for d, ax in enumerate(axarr.flatten()):
            ax.cla()
            ax.plot(datas[idx][:,d], label='real')
            paths = samps[:,:,d].T
            ax.plot(paths[0,:], c='grey', alpha=.4, label='gen')
            ax.plot(paths[1:,:], c='grey', alpha=.4)
            ax.legend()
            ylim = datas[idx][:,d].min(), datas[idx][:,d].max()
            ax.set_ylim(ylim)
            plt.draw()
        plt.pause(.01)

    def plot_multi():
        plot_data(5, axarr[:,0])
        plot_data(10, axarr[:,1])
        plot_data(12, axarr[:,2])

    for i in xrange(200):
        feed_dict = {seqs:datas_padded, seq_lengths: datas_lengths}
        res       = sess.run([train_step, cost], feed_dict=feed_dict)
        print "iter %d: "%i, res
        if i % 10 == 0:
            plot_multi()

