# -*- coding: utf-8 -*-

# news categories
NEWS_CATEGORIES = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
#                   0       1       2       3       4       5       6   7          8    9

NUM_CLASSES = len(NEWS_CATEGORIES)

VOCAB_SIZE = 5000
VOCAB_FILE = 'data/vocab.zh'

UNKNOWN_SYMBOL = '<UNK>'

RAW_TRAIN_PATH = 'data/cnews.train.txt'
#RAW_TRAIN_PATH = 'data/x.cnews.train.txt'
RAW_TEST_PATH = 'data/cnews.test.txt'
TRAIN_PATH = 'data/cnews.train.tfrecords'
TEST_PATH = 'data/cnews.test.tfrecords'

CKPT_DIR = 'ckpt'

# large batch, ex 200, does not work, I don't know why
BATCH_SIZE = 64

TEST_BATCH_SIZE = 1000

STEPS_PER_CKPT = 100

# shuffle size affects convergence greatly, it should be big enough
SHUFFLE_SIZE = 5000

SENTENCE_SIZE_MIN = 1
# documents: 50000 - 2500 length, 47391 remains; 1000 length, 34331 remains
SENTENCE_SIZE_MAX = 1000
#SENTENCE_SIZE_MAX = 600

SENTENCE_FILE = 'data/sentence.txt'

LEARNING_RATE = 1e-3

TRAIN_KEEP_PROB = 0.5
TEST_KEEP_PROB = 1.0

HIDDEN_SIZE = 64

EMBED_SIZE = 64
CONV_FILTER_NUM = 128
CONV_FILTER_KERNEL_SIZES = [2, 3, 4]

NUM_EPOCH = 5
