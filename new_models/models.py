from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from new_models import get_embeddings


def base_model(rnn_size,
               vocab_size,
               emb_size,
               vocab_path=None,
               emb_path=None):

    def model_fn(ctx_idx_data,
                 ans_idx_data):
        # set embedding layer
        with tf.variable_scope('embedding'):
            emb_W = get_embeddings(vocab_size=vocab_size,
                                   emb_size=emb_size,
                                   vocab_path=vocab_path,
                                   emb_path=emb_path)
            ctx_emb_data = tf.nn.embedding_lookup(params=emb_W,
                                                  ids=ctx_idx_data)
            ans_emb_data = tf.nn.embedding_lookup(params=emb_W,
                                                  ids=ans_idx_data)

        # set recurrent layer
        with tf.variable_scope('recurrent'):
            rnn_cell = tf.contrib.rnn.LSTMCell(num_units=rnn_size)
            ctx_enc_data = tf.nn.dynamic_rnn(cell=rnn_cell,
                                             inputs=ctx_emb_data,
                                             dtype=tf.float32)[-1].h
        with tf.variable_scope('recurrent'):
            rnn_cell = tf.contrib.rnn.LSTMCell(num_units=rnn_size, reuse=True)
            ans_enc_data = tf.nn.dynamic_rnn(cell=rnn_cell,
                                             inputs=ans_emb_data,
                                             dtype=tf.float32)[-1].h

        # set prediction layer
        with tf.variable_scope('prediction'):
            pred_W = tf.get_variable(name="W",
                                     shape=(rnn_size, rnn_size),
                                     initializer=tf.truncated_normal_initializer())
            ctx_enc_data = tf.matmul(ctx_enc_data, pred_W)
            sim_logits = tf.multiply(ctx_enc_data, ans_enc_data)
            sim_logits = tf.reduce_sum(sim_logits, axis=1, keep_dims=True)
            sim_probs = tf.sigmoid(sim_logits)

        return sim_logits, sim_probs

    return model_fn


def attn_model(ctx_idx_data,
               ans_idx_data,
               rnn_size,
               attn_size,
               vocab_size,
               emb_size,
               vocab_path=None,
               emb_path=None):

    # set embedding layer
    with tf.variable_scope('embedding'):
        emb_W = get_embeddings(vocab_size=vocab_size,
                               emb_size=emb_size,
                               vocab_path=vocab_path,
                               emb_path=emb_path)
        ctx_emb_data = tf.nn.embedding_lookup(params=emb_W,
                                              ids=ctx_idx_data)
        ans_emb_data = tf.nn.embedding_lookup(params=emb_W,
                                              ids=ans_idx_data)

    # set recurrent layer
    with tf.variable_scope('recurrent'):
        rnn_cell = tf.contrib.rnn.LSTMCell(num_units=rnn_size)
        ctx_enc_seq = tf.nn.dynamic_rnn(cell=rnn_cell,
                                        inputs=ctx_emb_data,
                                        dtype=tf.float32)[0]
    with tf.variable_scope('recurrent'):
        rnn_cell = tf.contrib.rnn.LSTMCell(num_units=rnn_size, reuse=True)
        ans_enc_data = tf.nn.dynamic_rnn(cell=rnn_cell,
                                         inputs=ans_emb_data,
                                         dtype=tf.float32)[-1].h

    # set attention layer
    with tf.variable_scope('attention'):
        attn_W = tf.get_variable(name="W",
                                 shape=(rnn_size, attn_size),
                                 initializer=tf.truncated_normal_initializer())
        attn_b = tf.get_variable(name="b",
                                 shape=(attn_size, ),
                                 initializer=tf.constant_initializer())
        attn_V = tf.get_variable(name="V",
                                 shape=(attn_size, 1),
                                 initializer=tf.truncated_normal_initializer())
        seq_len = ctx_enc_seq.shape[1].value
        attn_weight = tf.matmul(tf.reshape(ctx_enc_seq, (-1, rnn_size)), attn_W) + attn_b
        attn_weight = tf.tanh(attn_weight)
        attn_weight = tf.matmul(attn_weight, attn_V)
        attn_weight = tf.exp(tf.reshape(attn_weight, (-1, seq_len)))
        attn_weight = attn_weight/tf.reduce_sum(attn_weight, axis=1, keep_dims=True)
        ctx_enc_data = tf.reduce_sum(tf.expand_dims(attn_weight, axis=-1)*ctx_enc_seq, axis=1)

    # set prediction layer
    with tf.variable_scope('prediction'):
        pred_W = tf.get_variable(name="W",
                                 shape=(rnn_size, rnn_size),
                                 initializer=tf.truncated_normal_initializer())
        ctx_enc_data = tf.matmul(ctx_enc_data, pred_W)
        sim_logits = tf.multiply(ctx_enc_data, ans_enc_data)
        sim_logits = tf.reduce_sum(sim_logits, axis=1, keep_dims=True)
        sim_probs = tf.sigmoid(sim_logits)

    return sim_probs, sim_logits