import tensorflow as tf
import array
from collections import defaultdict
import numpy as np

# load voacb list from text
def load_vocab(filename):
    # open file
    with open(filename) as f:
        # read file
        vocab = f.read().splitlines()

    # init dict
    vocab_dict = defaultdict(int)

    # for each word
    for idx, word in enumerate(vocab):
        vocab_dict[word] = idx

    return vocab, vocab_dict

# load vector from text
def load_vectors(filename, vocab):
    word2idx = {}
    vectors = array.array('d')
    current_idx = 0
    # open file
    with open(filename, 'r') as f:
        # for each line
        for line in f:
            tokens = line.split(' ')
            word = tokens[0]
            entries = tokens[1:]

            # if in our vocab
            if not vocab or word in vocab:
                word2idx[word] = current_idx
                vectors.extend(float(x) for x in entries)
                current_idx += 1

        emb_size = len(entries)
        num_words = len(word2idx)

    tf.logging.info("Found {} out of {} vectors in Glove".format(num_words, len(vocab)))
    return np.array(vectors).reshape(num_words, emb_size), word2idx

# init embedding matrix
def initialize_embedding(total_vocab_dict,
                         exist_vocab_dict,
                         exist_emb_vectors):
    total_vocab_size = len(total_vocab_dict)
    emb_size = exist_emb_vectors.shape[-1]

    # random init embedding
    emb = np.random.uniform(-0.25, 0.25, (total_vocab_size, emb_size)).astype("float32")

    # load existing embedding
    for word, idx in exist_vocab_dict.items()
        emb[total_vocab_dict[word], :] = exist_emb_vectors[idx]

    return emb

# load embedding
def get_embeddings(vocab_size,
                   emb_size,
                   vocab_path=None,
                   emb_path=None):
    # check load path
    if vocab_path and emb_path:
        tf.logging.info("Loading Glove embeddings...")

        # load vocab
        vocab_array, vocab_dict = load_vocab(vocab_path)

        # load glove vectors
        glove_vectors, glove_dict = load_vectors(emb_path, vocab=set(vocab_array))

        # init emb
        initializer = initialize_embedding(total_vocab_dict=vocab_dict,
                                           exist_vocab_dict=glove_dict,
                                           exist_emb_vectors=glove_vectors)
    else:
        tf.logging.info("No glove/vocab path specificed, starting with random embeddings.")
        initializer = tf.random_uniform_initializer(-0.25, 0.25)

    return tf.get_variable(name="word_embeddings",
                           shape=[vocab_size, emb_size],
                           initializer=initializer)


