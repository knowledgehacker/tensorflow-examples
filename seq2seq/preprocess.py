import sys
import codecs
from enum import Enum
import collections
from operator import itemgetter
import config

UNKNOWN_SYMBOL = '<unk>'
START_OF_SEQUENCE = '<sos>'
END_OF_SEQUENCE = '<eos>'

Lang = Enum('Lang', ('SRC', 'TRG'))


# Build vocabulary
def build_vocab(input, vocab_size, vocab_file):
    print("build vocabulary: %s" % vocab_file)

    word_count = collections.Counter()
    with codecs.open(input, 'r', 'utf-8') as f:
        for line in f:
            for word in line.strip().split():
                word_count[word] += 1

    sorted_word_count = sorted(word_count.items(), key=itemgetter(1), reverse=True)
    sorted_words = [w[0] for w in sorted_word_count]
    sorted_words = [UNKNOWN_SYMBOL, START_OF_SEQUENCE, END_OF_SEQUENCE] + sorted_words
    if len(sorted_words) > vocab_size:
        sorted_words = sorted_words[:vocab_size]

    with codecs.open(vocab_file, 'w', 'utf-8') as f:
        for word in sorted_words:
            f.write(word + '\n')


# Build dataset, replace words with indices, with rare words replaced with UNKOWN_SYMBOL
def build_dataset(input, vocab_file, lang, output):
    print("build dataset: %s" % input)

    with codecs.open(vocab_file, 'r', 'utf-8') as f_vocab:
        vocab = [w.strip() for w in f_vocab.readlines()]
    word_index = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

    fin = codecs.open(input, 'r', 'utf-8')
    fout = codecs.open(output, 'w', 'utf-8')
    for line in fin:
        words = line.strip().split()
        indices = [str(get_index(word, word_index)) for word in words]
        #if lang == Lang.TRG:
        #    indices = indices + ['2']   # END_OF_SEQUENCE
        indices = indices + ['2']  # END_OF_SEQUENCE
        fout.write(' '.join(indices) + '\n')
    fin.close
    fout.close


def get_index(word, word_index):
    return word_index[word] if word in word_index else word_index[UNKNOWN_SYMBOL]


src_input = 'data/train.txt.en'
build_vocab(src_input, config.SRC_VOCAB_SIZE, config.SRC_VOCAB_FILE)
build_dataset(src_input, config.SRC_VOCAB_FILE, Lang.SRC, config.TRAIN_SRC_PATH)


trg_input = 'data/train.txt.zh'
build_vocab(trg_input, config.TRG_VOCAB_SIZE, config.TRG_VOCAB_FILE)
build_dataset(trg_input, config.TRG_VOCAB_FILE, Lang.TRG, config.TRAIN_TRG_PATH)
