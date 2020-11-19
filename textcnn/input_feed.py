# -*- coding: utf-8 -*-

import tensorflow.compat.v1 as tf

import config

tf.disable_v2_behavior()


def init_next_batch(input, test=False):
    with tf.device('/cpu:0'):
        dataset = create_dataset(input, test)
        iterator = dataset.make_initializable_iterator()
        dataset_init_op = iterator.make_initializer(dataset, name='dataset_init')
        # must have different name with the one in the graph loaded from trained model to avoid name conflict?
        get_next_op = iterator.get_next(name='get_next')

    return dataset_init_op, get_next_op


def create_dataset(input, test):
    dataset = tf.data.TFRecordDataset(input)
    dataset = dataset.map(parse)
    dataset = dataset.map(lambda content, label: (content, tf.size(content), label))
    dataset = dataset.filter(lambda content, size, label: tf.logical_and(tf.greater(size, 1),
                                                                        tf.less_equal(size, config.SENTENCE_MAX_LEN)))
    dataset = dataset.map(lambda content, size, label: (content, label))

    if not test:
        dataset = dataset.shuffle(config.SHUFFLE_SIZE)

    # TODO: how padding here affects training?
    padded_shapes = (
        (tf.TensorShape([config.SENTENCE_MAX_LEN]),
         tf.TensorShape([config.NUM_CLASSES])))
    batched_dataset = dataset.padded_batch(config.BATCH_SIZE, padded_shapes)

    return batched_dataset


def parse(record):
    example = tf.parse_single_example(
        record,
        features={
            'indices': tf.VarLenFeature(tf.int64),
            'label': tf.FixedLenFeature([config.NUM_CLASSES], tf.float32)
        })

    # represent 'indices' as a dense tensor, used as a feed to tf.nn.embedding_lookup. refer to seq2seq model
    return tf.sparse_tensor_to_dense(example['indices']), example['label']
