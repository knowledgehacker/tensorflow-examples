# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_ranking as tfr
from config import FEATURE_NUM, MAX_DOC_NUM, BATCH_SIZE, SHUFFLE_SIZE


def create_dataset(input, test):
    dataset = tf.data.Dataset.from_generator(
        tfr.data.libsvm_generator(input, FEATURE_NUM, MAX_DOC_NUM),
        output_types=(
            {str(k): tf.float32 for k in range(1, FEATURE_NUM + 1)},
            tf.float32
        ),
        output_shapes=(
            {str(k): tf.TensorShape([MAX_DOC_NUM, 1])
                for k in range(1, FEATURE_NUM + 1)},
            tf.TensorShape([MAX_DOC_NUM])
        )
    )

    if not test:
        dataset = dataset.shuffle(SHUFFLE_SIZE)

    batched_dataset = dataset.batch(BATCH_SIZE)

    return batched_dataset
