"""
Implements a deep generative LSTM for multi-dimensional, variable-length
sequences.   For a single sequence, the data generating process is

  Z   ~ Normal(0, I)           (H-dimensional latent variable )
  M   = self-feeding-lstm(Z)   LSTM with hidden dimension H, using Z as its initial state
  Y   ~ N(M, I)                The LSTM outputs a mean trajectory (deteterministically)
                               with noisy observations about the mean

The inference network is also an LSTM, with input Y, and latent dimension P

  O_1, H_1, ..., O_T, H_T = input-lstm(Y)
  z_mu, z_logsigmasq      = MLP(H_T, phi), MLP(H_T, phi)
  Z | Y                   ~ N(z_mu, np.exp(z_logsigmasq))


The following functions make the generative and inference networks

    - decoder: Z (latent), T (known)      => Y (observation)
    - encoder: Y (observation), T (known) => mu_z, sigmasq_z
    - vlb    : MC estimate of the variational lower bound

"""
import tensorflow as tf
import numpy as np
from tensorflow import nn
from vae.vae import kl_to_prior

def make_lstm_decoder(num_hidden, data_dim, max_length):

    with tf.variable_scope("lstm_decode"):
        # set up lstm cell that computes (output, state) pairs
        cell = nn.rnn_cell.BasicLSTMCell(num_hidden, state_is_tuple=True)
        #cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)
        #cell = rnn_cell.MultiRNNCell([cell] * num_layers)
        cell = nn.rnn_cell.OutputProjectionWrapper(cell, output_size=data_dim)

    def decode(latent_vars):
        """
            Decode a Z dimensional random variable into a sequence with
            a fixed length

        Args:
            - Z: tf.placeholder/tensor with dimensions [batch_size, num_hidden]

        Returns:
            - output: tf.tensor with dimensions [batch_size, max_length, data_dim]
                      output of sequence
        """
        # set up the loop for sequence generation)
        i         = tf.constant(0)
        cond      = lambda i, _0, _1, _2: tf.less(i, max_length)

        # create lstm state tuple --- forget c gate first, then hidden state Z
        cstate      = tf.ones_like(latent_vars, name="decode_cstate")
        state_tuple = nn.rnn_cell.LSTMStateTuple(cstate, latent_vars)

        # output tensor array --- keeps cell output for each time step
        output    = tf.zeros(shape = [tf.shape(latent_vars)[0], data_dim],
                             name  = "decoder_output")
        output_ta = tf.TensorArray(dtype=output.dtype, size=max_length,
                                   tensor_array_name="decode_output_ta")

        def loop_body(i, output, state_tuple, output_ta):
            # step the cell forward
            output, state_tuple = cell(output, state_tuple,
                                       scope="lstm_decode")
            # save output
            output_ta = output_ta.write(i, output)
            return i+1, output, state_tuple, output_ta

        i_last, o_last, h_last, output_ta = \
            tf.while_loop(cond, loop_body,
                          loop_vars           = [i, output, state_tuple, output_ta],
                          parallel_iterations = 10,
                          back_prop           = True,
                          swap_memory         = False,
                          name                = "decoder")

        # ensure output is [batch_size, max_length, data_dim]
        output_seq = tf.transpose(output_ta.pack(), [1, 0, 2])
        return output_seq
    return decode


def make_lstm_encoder(num_hidden, data_dim, encoder_hidden, max_length):
    """ creates an encoder function that takes a sequence of varying length
    and maps it to a mean and variance in a space of dimension "num_hidden"

    This encoder function itself is an LSTM with some hidden dimension
    (encoder_hidden) that steps over the sequence, where the last state
    is mapped to a mean and log_variance.
    """
    # set up lstm cell that computes (output, state) pairs
    with tf.variable_scope("lstm_encode"):
        cell = nn.rnn_cell.BasicLSTMCell(encoder_hidden, state_is_tuple=True)
        #cell = nn.rnn_cell.OutputProjectionWrapper(cell, output_size=num_hidden)
    Wmu  = tf.Variable(tf.truncated_normal(shape=[encoder_hidden,num_hidden]),
                                              name='Wmu')
    bmu  = tf.Variable(tf.random_normal(shape=[num_hidden]), name='bmu')
    Wsig = tf.Variable(tf.truncated_normal(shape=[encoder_hidden,num_hidden]),
                                              name='Wsig')
    bsig = tf.Variable(tf.random_normal(shape=[num_hidden]), name='bsig')

    def encode(seqs, seq_lengths):
        """ maps data observation (seqs) to posterior distribution over
            latent variables

        Args:
            - seqs: tensor [batch_size, max_length, data_dim] of observed sequences
            - max_length: tensor [batch_size] of sequence lengths

        """
        # set up the loop for sequence generation)
        i         = tf.constant(0)
        cond      = lambda i, _0, _1, _2: tf.less(i, max_length)

        # create lstm state tuple --- forget c gate first, then hidden state Z
        cstate      = tf.ones(shape =[tf.shape(seqs)[0], encoder_hidden], name="encode_cstate")
        hstate      = tf.zeros(shape=[tf.shape(seqs)[0], encoder_hidden], name="encode_hstate")
        state_tuple = nn.rnn_cell.LSTMStateTuple(cstate, hstate)

        # unpack inputs (seqs) into a tensor array for easy indexing
        input_ta = tf.TensorArray(dtype=seqs.dtype, size=max_length,
                                  tensor_array_name="encode_seq_input")
        input_ta = input_ta.unpack(tf.transpose(seqs, [1, 0, 2]))

        # output tensor array --- keeps cell output for each time step
        output = tf.zeros(shape = [tf.shape(seqs)[0], data_dim],
                          name  = "decoder_output")
        h_ta   = tf.TensorArray(dtype=hstate.dtype, size=max_length,
                                tensor_array_name="decode_h_ta")

        def loop_body(i, output, state_tuple, h_ta):
            # step the cell forward
            output, state_tuple = cell(input_ta.read(i), state_tuple,
                                       scope="lstm_encode")
            h_ta = h_ta.write(i, state_tuple[1])
            return i+1, output, state_tuple, h_ta

        i_last, o_last, s_last, h_ta = \
            tf.while_loop(cond, loop_body,
                          loop_vars           = [i, output, state_tuple, h_ta],
                          parallel_iterations = 10,
                          back_prop           = True,
                          swap_memory         = False,
                          name                = "encoder")

        # pack up all states into a tensor [ max_length, num_batches, h_dim ]
        hta        = tf.transpose(h_ta.pack(), [1, 0, 2])
        hta_flat   = tf.reshape(hta, [-1, encoder_hidden])
        batch_size = tf.shape(seqs)[0]
        idx_flat   = tf.range(0, batch_size) * max_length + (seq_lengths-1)
        hend       = tf.gather(hta_flat, idx_flat)

        # take the last states, and project it down to a mean and variance
        mu         = tf.matmul(hend, Wmu) + bmu
        logsigmasq = tf.matmul(hend, Wsig) + bsig
        return mu, logsigmasq

    return encode


def sequence_loglike(X, M, seq_lengths):
    """
        Gaussian log likelihood for variable length sequences.
    """
    # make the mask (for varying sequence lengths)
    mask  = tf.cast(make_mask(tf.shape(X)[0], tf.shape(X)[1], seq_lengths), dtype=tf.float32)
    D = tf.shape(X)[2]
    mask  = tf.tile(tf.expand_dims(mask, 2), tf.pack([1, 1, D]))

    # element-wise diffs
    distsq = tf.reduce_sum(tf.mul((X - M) ** 2, mask), [2])
    lls    = -.5*tf.reduce_sum( (np.log(2.*np.pi) + 0.) + distsq / 1.,
                                reduction_indices=1)
    return lls


def make_mask(batch_size, max_length, seq_lengths):
    lengths_transposed = tf.expand_dims(seq_lengths, 1)
    lengths_tiled = tf.tile(lengths_transposed, [1, max_length])
    trange        = tf.cast(tf.range(0, max_length, 1), dtype=np.int32)
    trange_row    = tf.expand_dims(trange, 0)
    trange_tiled  = tf.tile(trange_row, [batch_size, 1])
    mask          = tf.less(trange_tiled, lengths_tiled)
    return mask


def make_vae_objective(encode, decode, zdim):
    def vlb(datas, L, seq_lengths):
        """ variational lower bound
            Args:
              - X : Nbatch x dimx data matrix
              - L : number of samples for monte carlo estimate
        """
        M = tf.shape(datas)[0]
        def sample_z(mu, log_sigmasq):
            eps = tf.random_normal(shape=[M, zdim], dtype=tf.float32)
            return mu + tf.exp(0.5 * log_sigmasq) * eps

        mu, log_sigmasq = encode(datas, seq_lengths)
        M = decode(sample_z(mu, log_sigmasq))

        logpxz = sum(sequence_loglike(datas, M, seq_lengths)
                     for l in xrange(L)) / float(L)

        minibatch_val = -kl_to_prior(mu, log_sigmasq) + logpxz
        return minibatch_val

    return vlb

