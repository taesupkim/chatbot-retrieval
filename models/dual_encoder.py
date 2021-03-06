import tensorflow as tf
import numpy as np
from models import helpers

FLAGS = tf.flags.FLAGS

EOT = 2

def get_embeddings(hparams):
  if hparams.glove_path and hparams.vocab_path:
    tf.logging.info("Loading Glove embeddings...")
    vocab_array, vocab_dict = helpers.load_vocab(hparams.vocab_path)
    glove_vectors, glove_dict = helpers.load_glove_vectors(hparams.glove_path, vocab=set(vocab_array))
    initializer = helpers.build_initial_embedding_matrix(vocab_dict, glove_dict, glove_vectors, hparams.embedding_dim)
  else:
    tf.logging.info("No glove/vocab path specificed, starting with random embeddings.")
    initializer = tf.random_uniform_initializer(-0.25, 0.25)

  return tf.get_variable(
    "word_embeddings",
    shape=[hparams.vocab_size, hparams.embedding_dim],
    initializer=initializer)


def dual_encoder_model(
    hparams,
    mode,
    context,
    context_len,
    utterance,
    utterance_len,
    targets):

  # Initialize embedidngs randomly or with pre-trained vectors if available
  embeddings_W = get_embeddings(hparams)

  # Embed the context and the utterance
  context_embedded = tf.nn.embedding_lookup(embeddings_W,
                                            context,
                                            name="embed_context")
  utterance_embedded = tf.nn.embedding_lookup(embeddings_W,
                                              utterance,
                                              name="embed_utterance")


  # Build the RNN
  with tf.variable_scope("rnn") as vs:
    # We use an LSTM Cell
    cell = tf.contrib.rnn.LSTMCell(hparams.rnn_dim,
                                   forget_bias=2.0,
                                   use_peepholes=True,
                                   state_is_tuple=True)

    # Run the utterance and context through the RNN
    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell,
                                                tf.concat(0, [context_embedded, utterance_embedded]),
                                                sequence_length=tf.concat(0, [context_len, utterance_len]),
                                                dtype=tf.float32)
    encoding_context, encoding_utterance = tf.split(0, 2, rnn_states.h)

  with tf.variable_scope("prediction") as vs:
    M = tf.get_variable("M",
                        shape=[hparams.rnn_dim, hparams.rnn_dim],
                        initializer=tf.truncated_normal_initializer())

    # "Predict" a  response: c * M
    generated_response = tf.matmul(encoding_context, M)

    logits = tf.multiply(generated_response, encoding_utterance)
    logits = tf.reduce_sum(logits, axis=1, keep_dims=True)

    # Apply sigmoid to convert logits to probabilities
    probs = tf.sigmoid(logits)

    if mode == tf.contrib.learn.ModeKeys.INFER:
      return probs, None

    # Calculate the binary cross-entropy loss
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.to_float(targets))

  # Mean loss across the batch of examples
  mean_loss = tf.reduce_mean(losses, name="mean_loss")
  return probs, mean_loss

def attention_model(
    hparams,
    mode,
    context,
    context_len,
    utterance,
    utterance_len,
    targets):

  # Initialize embedidngs randomly or with pre-trained vectors if available
  embeddings_W = get_embeddings(hparams)

  # Embed the context and the utterance
  context_embedded = tf.nn.embedding_lookup(embeddings_W,
                                            context,
                                            name="embed_context")
  utterance_embedded = tf.nn.embedding_lookup(embeddings_W,
                                              utterance,
                                              name="embed_utterance")


  # Build the RNN
  with tf.variable_scope("rnn") as vs:
    # We use an LSTM Cell
    cell = tf.contrib.rnn.LSTMCell(hparams.rnn_dim,
                                   forget_bias=2.0,
                                   use_peepholes=True,
                                   state_is_tuple=True)

    # Run the utterance and context through the RNN
    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell,
                                                tf.concat(0, [context_embedded, utterance_embedded]),
                                                sequence_length=tf.concat(0, [context_len, utterance_len]),
                                                dtype=tf.float32)
    _, encoding_utterance = tf.split(0, 2, rnn_states.h)
    encoding_context_seq, _ = tf.split(0, 2, rnn_outputs)

  with tf.variable_scope("attn") as vs:
    encoding_context_select = tf.boolean_mask(encoding_context_seq, tf.equal(context, EOT))

    attn_W = tf.get_variable(name="W",
                             shape=(hparams.rnn_dim, hparams.rnn_dim/2),
                             initializer=tf.truncated_normal_initializer())
    attn_b = tf.get_variable(name="b",
                             shape=(hparams.rnn_dim/2, ),
                             initializer=tf.constant_initializer())
    attn_V = tf.get_variable(name="V",
                             shape=(hparams.rnn_dim/2, 1),
                             initializer=tf.truncated_normal_initializer())

    seq_len = encoding_context_select.shape[1].value
    attn_weight = tf.matmul(tf.reshape(encoding_context_select, (-1, hparams.rnn_dim)), attn_W) + attn_b
    attn_weight = tf.tanh(attn_weight)
    attn_weight = tf.matmul(attn_weight, attn_V)
    attn_weight = tf.exp(tf.reshape(attn_weight, (-1, seq_len)))
    attn_weight = attn_weight/tf.reduce_sum(attn_weight, axis=1, keep_dims=True)
    encoding_context = tf.reduce_sum(tf.expand_dims(attn_weight, axis=-1)*encoding_context_select, axis=1)

  with tf.variable_scope("prediction") as vs:
    M = tf.get_variable("M",
                        shape=[hparams.rnn_dim, hparams.rnn_dim],
                        initializer=tf.truncated_normal_initializer())

    # "Predict" a  response: c * M
    generated_response = tf.matmul(encoding_context, M)

    logits = tf.multiply(generated_response, encoding_utterance)
    logits = tf.reduce_sum(logits, axis=1, keep_dims=True)

    # Apply sigmoid to convert logits to probabilities
    probs = tf.sigmoid(logits)

    if mode == tf.contrib.learn.ModeKeys.INFER:
      return probs, None

    # Calculate the binary cross-entropy loss
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.to_float(targets))

  # Mean loss across the batch of examples
  mean_loss = tf.reduce_mean(losses, name="mean_loss")
  return probs, mean_loss
