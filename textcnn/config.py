# -*- coding: utf-8 -*-

# news categories
NEWS_CATEGORIES = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
#                   0       1       2       3       4       5       6   7          8    9

NUM_CLASSES = len(NEWS_CATEGORIES)

VOCAB_SIZE = 5000
VOCAB_FILE = 'data/vocab.zh'

UNKNOWN_SYMBOL = '<UNK>'

RAW_TRAIN_PATH = 'data/cnews.train.txt'
RAW_TEST_PATH = 'data/cnews.test.txt'
TRAIN_PATH = 'data/cnews.train.tfrecords'
TEST_PATH = 'data/cnews.test.tfrecords'

CKPT_DIR = 'ckpt'

STEPS_PER_CKPT = 50

BATCH_SIZE = 64

# shuffle size affects convergence greatly, it should be big enough
SHUFFLE_SIZE = 10000

# about %5 of all documents(50000) in train set is filtered out
SENTENCE_MAX_LEN = 1000

EMBED_SIZE = 64

HIDDEN_SIZE = 64

TRAIN_KEEP_PROB = 0.5
TEST_KEEP_PROB = 1.0

CONV_FILTER_NUM = 64
CONV_FILTER_KERNEL_SIZES = [2, 3, 4]

NUM_EPOCH = 3
