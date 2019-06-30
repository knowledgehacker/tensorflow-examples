import config
import tensorflow as tf

MAX_LEN = 50


def create_src_trg_dataset(src_path, trg_path):
    src_data = create_dataset(src_path)
    trg_data = create_dataset(trg_path)
    dataset = tf.data.Dataset.zip((src_data, trg_data))

    def filter_length(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        src_len_ok = tf.logical_and(
            tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
        trg_len_ok = tf.logical_and(
            tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))
        return tf.logical_and(src_len_ok, trg_len_ok)
    dataset = dataset.filter(filter_length)

    def create_trg_input(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[1], trg_label[:-1]], axis=0)
        return (src_input, src_len), (trg_input, trg_len, trg_label)
    dataset = dataset.map(create_trg_input)

    dataset = dataset.shuffle(10000)

    padded_shapes = (
        (tf.TensorShape([None]),
         tf.TensorShape([])),
        (tf.TensorShape([None]),
         tf.TensorShape([]),
         tf.TensorShape([None])))
    batched_dataset = dataset.padded_batch(config.BATCH_SIZE, padded_shapes)

    return batched_dataset


def create_dataset(input):
    dataset = tf.data.TextLineDataset(input)
    dataset = dataset.map(lambda line: tf.string_split([line]).values)
    dataset = dataset.map(lambda words: tf.string_to_number(words, tf.int32))
    # ingore few InvalidArgumentError thrown by tf.string_to_number
    dataset = dataset.apply(tf.contrib.data.ignore_errors())
    dataset = dataset.map(lambda indices: (indices, tf.size(indices)))

    return dataset


'''
def s2i(words):
    try:
        idx = tf.string_to_number(words, tf.int32)
    #except tf.errors.InvalidArgumentError as e:
    except:
        print("words: %s" % words)
        #print("s2i failed - [%s]" % e)

    return idx
'''