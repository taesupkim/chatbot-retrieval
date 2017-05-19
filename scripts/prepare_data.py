from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import functools
import tensorflow as tf

tf.flags.DEFINE_integer("max_ctx_len", 160, "Maximum Context Length")
tf.flags.DEFINE_integer("max_ans_len", 160, "Maximum Answer Length")

tf.flags.DEFINE_integer("min_word_frequency", 5, "Minimum frequency of words in the vocabulary")
tf.flags.DEFINE_integer("max_sentence_len", 160, "Maximum Sentence Length")
tf.flags.DEFINE_string("input_dir", os.path.abspath("./data"), "Input directory containing original CSV data files (default = './data')")
tf.flags.DEFINE_string("output_dir", os.path.abspath("./data"), "Output directory for TFrEcord files (default = './data')")

FLAGS = tf.flags.FLAGS

TRAIN_PATH = os.path.join(FLAGS.input_dir, "train.csv")
VALIDATION_PATH = os.path.join(FLAGS.input_dir, "valid.csv")
TEST_PATH = os.path.join(FLAGS.input_dir, "test.csv")


# tokenizer based on space ' '
def tokenizer_fn(iterator):
    return (x.split(" ") for x in iterator)


# get csv iterator without header
def create_csv_iter(filename):
    with open(filename) as csvfile:
        # get reader
        reader = csv.reader(csvfile)

        # skip header
        next(reader)

        # iterator
        for row in reader:
            yield row


# create vocab from input
def create_vocab(input_iter,
                 min_frequency):
    # create vocab processor
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length=FLAGS.max_ctx_len,
                                                                         min_frequency=min_frequency,
                                                                         tokenizer_fn=tokenizer_fn)

    # collection vocab
    vocab_processor.fit(input_iter)
    return vocab_processor


# transform sentence into index
def transform_sentence(sequence,
                       vocab_processor):
    return next(vocab_processor.transform([sequence])).tolist()



def create_text_sequence_feature(fl,
                                 sentence,
                                 vocab):
    """
      Writes a sentence to FeatureList protocol buffer
    """
    sentence_transformed = transform_sentence(sentence, vocab)
    for word_id in sentence_transformed:
        fl.feature.add().int64_list.value.extend([word_id])
    return fl


# create single train example
def create_example_train(row_data,
                         vocab_process):
    """
      Creates a training example for the Ubuntu Dialog Corpus dataset.
      Returnsthe a tensorflow.Example Protocol Buffer object.
    """
    context, utterance, label = row_data
    context_transformed = transform_sentence(context, vocab_process)
    utterance_transformed = transform_sentence(utterance, vocab_process)
    context_len = len(next(vocab_process._tokenizer([context])))
    utterance_len = len(next(vocab_process._tokenizer([utterance])))
    label = int(float(label))

    # New Example
    example = tf.train.Example()
    example.features.feature["context"].int64_list.value.extend(context_transformed)
    example.features.feature["utterance"].int64_list.value.extend(utterance_transformed)
    example.features.feature["context_len"].int64_list.value.extend([context_len])
    example.features.feature["utterance_len"].int64_list.value.extend([utterance_len])
    example.features.feature["label"].int64_list.value.extend([label])
    return example


def create_example_test(row, vocab_process):
    """
      Creates a test/validation example for the Ubuntu Dialog Corpus dataset.
      Returnsthe a tensorflow.Example Protocol Buffer object.
    """
    context, utterance = row[:2]
    distractors = row[2:]
    context_len = len(next(vocab_process._tokenizer([context])))
    utterance_len = len(next(vocab_process._tokenizer([utterance])))
    context_transformed = transform_sentence(context, vocab_process)
    utterance_transformed = transform_sentence(utterance, vocab_process)

    # New Example
    example = tf.train.Example()
    example.features.feature["context"].int64_list.value.extend(context_transformed)
    example.features.feature["utterance"].int64_list.value.extend(utterance_transformed)
    example.features.feature["context_len"].int64_list.value.extend([context_len])
    example.features.feature["utterance_len"].int64_list.value.extend([utterance_len])

    # Distractor sequences
    for i, distractor in enumerate(distractors):
        dis_key = "distractor_{}".format(i)
        dis_len_key = "distractor_{}_len".format(i)
        # Distractor Length Feature
        dis_len = len(next(vocab_process._tokenizer([distractor])))
        example.features.feature[dis_len_key].int64_list.value.extend([dis_len])
        # Distractor Text Feature
        dis_transformed = transform_sentence(distractor, vocab_process)
        example.features.feature[dis_key].int64_list.value.extend(dis_transformed)
    return example


def create_tfrecords_file(input_filename, output_filename, example_fn):
    """
      Creates a TFRecords file for the given input data and
      example transofmration function
    """
    writer = tf.python_io.TFRecordWriter(output_filename)
    print("Creating TFRecords file at {}...".format(output_filename))

    for i, row in enumerate(create_csv_iter(input_filename)):
        x = example_fn(row)
        writer.write(x.SerializeToString())
    writer.close()
    print("Wrote to {}".format(output_filename))


def write_vocabulary(vocab_processor, outfile):
    """
      Writes the vocabulary to a file, one word per line.
    """
    vocab_size = len(vocab_processor.vocabulary_)
    with open(outfile, "w") as vocabfile:
        for id in range(vocab_size):
            word =  vocab_processor.vocabulary_._reverse_mapping[id]
            vocabfile.write(word + "\n")
    print("Saved vocabulary to {}".format(outfile))


if __name__ == "__main__":
    print("Creating vocabulary...")
    input_iter = create_csv_iter(TRAIN_PATH)
    input_iter = (x[0] + " " + x[1] for x in input_iter)
    vocab = create_vocab(input_iter, min_frequency=FLAGS.min_word_frequency)
    print("Total vocabulary size: {}".format(len(vocab.vocabulary_)))

    # Create vocabulary.txt file
    write_vocabulary(vocab,
                     os.path.join(FLAGS.output_dir, "vocabulary.txt"))

    # Save vocab processor
    vocab.save(os.path.join(FLAGS.output_dir, "vocab_processor.bin"))

    # Create validation.tfrecords
    create_tfrecords_file(input_filename=VALIDATION_PATH,
                          output_filename=os.path.join(FLAGS.output_dir, "validation.tfrecords"),
                          example_fn=functools.partial(create_example_test, vocab=vocab))

    # Create test.tfrecords
    create_tfrecords_file(input_filename=TEST_PATH,
                          output_filename=os.path.join(FLAGS.output_dir, "test.tfrecords"),
                          example_fn=functools.partial(create_example_test, vocab=vocab))

    # Create train.tfrecords
    create_tfrecords_file(input_filename=TRAIN_PATH,
                          output_filename=os.path.join(FLAGS.output_dir, "train.tfrecords"),
                          example_fn=functools.partial(create_example_train, vocab=vocab))
