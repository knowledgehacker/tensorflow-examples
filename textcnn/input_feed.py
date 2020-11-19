# -*- coding: utf-8 -*-

import tensorflow.compat.v1 as tf

import config

tf.disable_v2_behavior()


def init_next_batch(input, sentence_max_size, test=False):
    with tf.device('/cpu:0'):
        dataset = create_dataset(input, sentence_max_size, test)
        iterator = dataset.make_initializable_iterator()
        dataset_init_op = iterator.make_initializer(dataset, name='dataset_init')
        # must have different name with the one in the graph loaded from trained model to avoid name conflict?
        get_next_op = iterator.get_next(name='get_next')

    return dataset_init_op, get_next_op


"""
def create_dataset(input, sentence_max_size, test):
    dataset = tf.data.TFRecordDataset(input)

    if not test:
        dataset = dataset.shuffle(config.SHUFFLE_SIZE)
        dataset = dataset.batch(config.BATCH_SIZE)
    else:
        dataset = dataset.batch(config.TEST_BATCH_SIZE)

    dataset = dataset.map(parse_batch)
    dataset = dataset.map(lambda content, label: (content, tf.size(content), label))
    dataset = dataset.filter(lambda content, size, label:
                             tf.logical_and(tf.greater(size, 1), tf.less_equal(size, sentence_max_size)))
    dataset = dataset.map(lambda content, size, label: (content, label))

    return dataset


def parse_batch(records):
    examples = tf.io.parse_example(
        records,
        features={
            'indices': tf.VarLenFeature(tf.int64),
            'label': tf.FixedLenFeature([config.NUM_CLASSES], tf.int64)
            #'label': tf.FixedLenFeature([config.NUM_CLASSES], tf.float32)
        })

    # represent 'indices' as a dense tensor, used as a feed to tf.nn.embedding_lookup. refer to seq2seq model
    return tf.sparse_tensor_to_dense(examples['indices']), examples['label']
"""


def create_dataset(input, sentence_size_max, test):
    dataset = tf.data.TFRecordDataset(input)

    dataset = dataset.map(parse)
    dataset = dataset.map(lambda content, label: (content, tf.size(content), label))
    dataset = dataset.filter(lambda content, size, label:
                             tf.logical_and(tf.greater(size, config.SENTENCE_SIZE_MIN),
                                            tf.less_equal(size, sentence_size_max)))
    dataset = dataset.map(lambda content, size, label: (content, label))

    # TODO: how padding here affects training?
    """
    Padding with 0, which is the same as the index of config.UNKNOWN_SYMBOL,
    will this cause train result incorrect?
    Considering the case max pooling after convolutions selects index 0, which is the padding 0,
    will the model learns the padding 0 is insignificant after convolutions?
    """
    padded_shapes = (
        (tf.TensorShape([sentence_size_max]),
         tf.TensorShape([config.NUM_CLASSES])))
    if not test:
        dataset = dataset.shuffle(config.SHUFFLE_SIZE)
        dataset = dataset.padded_batch(config.BATCH_SIZE, padded_shapes)
    else:
        dataset = dataset.padded_batch(config.TEST_BATCH_SIZE, padded_shapes)

    return dataset


def parse(record):
    example = tf.io.parse_single_example(
        record,
        features={
            'indices': tf.VarLenFeature(tf.int64),
            'label': tf.FixedLenFeature([config.NUM_CLASSES], tf.float32)
        })

    # represent 'indices' as a dense tensor, used as a feed to tf.nn.embedding_lookup. refer to seq2seq model
    return tf.sparse_tensor_to_dense(example['indices']), example['label']
