# -*- coding: utf-8 -*-

import codecs
import collections
from operator import itemgetter
import tensorflow as tf
import config
from utils import current_time


# Build vocabulary
def build_vocab(input, vocab_size, vocab_file):
    print(current_time(), "build vocabulary: %s starts..." % vocab_file)

    word_count = collections.Counter()
    with codecs.open(input, 'r', 'utf-8') as f:
        for line in f:
            splits = line.strip().split('\t')
            if len(splits) == 2:
                content = list(splits[1])   # split a sentence into words
                for word in content:
                    word_count[word] += 1

    sorted_word_count = sorted(word_count.items(), key=itemgetter(1), reverse=True)
    sorted_words = [config.UNKNOWN_SYMBOL] + [w[0] for w in sorted_word_count]
    print(current_time(), "word num: %d" % len(sorted_words))
    if len(sorted_words) > vocab_size:
        sorted_words = sorted_words[:vocab_size]

    with codecs.open(vocab_file, 'w', 'utf-8') as f:
        for word in sorted_words:
            f.write(word + '\n')

    print(current_time(), "build vocabulary: %s finishes..." % vocab_file)


# Build dataset, replace words with indices, with rare words replaced with UNKOWN_SYMBOL
def build_dataset(input, vocab_file, output):
    print(current_time(), "build dataset: %s starts..." % input)

    cate_to_index = build_cate_to_index()

    with codecs.open(vocab_file, 'r', 'utf-8') as f_vocab:
        vocab = [w.strip() for w in f_vocab.readlines()]
    word_index = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

    fin = codecs.open(input, 'r', 'utf-8')
    writer = tf.python_io.TFRecordWriter(output)
    #writer = tf.python_io.TFRecordWriter(output, options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP))
    for line in fin:
        splits = line.strip().split('\t')
        if len(splits) == 2:
            content = list(splits[1])
            indices = [get_index(word, word_index) for word in content]
            label = one_hot_encode(cate_to_index[splits[0]], config.NUM_CLASSES)
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'indices': int64_feature(indices),
                    'label': int64_feature(label),
                }))
            writer.write(example.SerializeToString())
    fin.close
    writer.close

    print(current_time(), "build dataset: %s finishes..." % input)


def build_cate_to_index():
    news_cates = [new_category.decode('utf-8') for new_category in config.NEWS_CATEGORIES]  # 'utf-8'
    cate_to_index = dict(zip(news_cates, range(config.NUM_CLASSES)))

    return cate_to_index


def get_index(word, word_index):
    return word_index[word] if word in word_index else word_index[config.UNKNOWN_SYMBOL]


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def one_hot_encode(index, num):
    ohe = [0 for i in range(num)]
    ohe[index] = 1

    return ohe


build_vocab(config.RAW_TRAIN_PATH, config.VOCAB_SIZE, config.VOCAB_FILE)
build_dataset(config.RAW_TRAIN_PATH, config.VOCAB_FILE, config.TRAIN_PATH)
build_dataset(config.RAW_TEST_PATH, config.VOCAB_FILE, config.TEST_PATH)
