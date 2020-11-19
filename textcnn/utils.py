import time


def current_time():
    return time.strftime('%H:%M:%S', time.localtime(time.time()))


def save(sentence_file, sentence_max_size):
    with open(sentence_file, 'w') as fout:
        fout.write(str(sentence_max_size))


def restore(sentence_file):
    with open(sentence_file, 'r') as fin:
        sentence_max_size = int(fin.readline().strip())

    return sentence_max_size
