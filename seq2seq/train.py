# -*- coding: utf-8 -*-
'''https://github.com/caicloud/tensorflow-tutorial/tree/master/Deep_Learning_with_TensorFlow/1.4.0/Chapter09'''

import sys
import codecs
import tensorflow as tf
import config
from input_feed import create_src_trg_dataset
from utils import current_time

EMBED_SIZE = 1024

HIDDEN_SIZE = config.HIDDEN_SIZE
NUM_LAYERS = config.NUM_LAYERS

SHARE_EMBED_AND_SOFTMAX = config.SHARE_EMBED_AND_SOFTMAX

KEEP_PROB = 0.8
MAX_GRAD_NORM = 5


class Seq2seqModel(object):
    def __init__(self):
        self.src_embeddings = tf.get_variable('src_embed', [config.SRC_VOCAB_SIZE, EMBED_SIZE], dtype=tf.float32)
        self.encode_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])

        self.trg_embeddings = tf.get_variable('trg_embed', [config.TRG_VOCAB_SIZE, EMBED_SIZE], dtype=tf.float32)
        self.decode_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])

        if SHARE_EMBED_AND_SOFTMAX:
            self.softmax_weights = tf.transpose(self.trg_embeddings)
        else:
            self.softmax_weights = tf.get_variable('softmax_weights', [HIDDEN_SIZE, config.TRG_VOCAB_SIZE], dtype=tf.float32)
        self.softmax_biases = tf.get_variable('softmax_biases', [config.TRG_VOCAB_SIZE], dtype=tf.float32)

    def forward(self, src_input, src_len, trg_input, trg_len, trg_labels):
        src_embed = tf.nn.embedding_lookup(self.src_embeddings, src_input)
        tf.nn.dropout(src_embed, KEEP_PROB)
        with tf.variable_scope('encoder'):
            _, encode_state = tf.nn.dynamic_rnn(self.encode_cell, src_embed, src_len, dtype=tf.float32)

        trg_embed = tf.nn.embedding_lookup(self.trg_embeddings, trg_input)
        tf.nn.dropout(trg_embed, KEEP_PROB)
        with tf.variable_scope('decoder'):
            decode_outputs, _ = tf.nn.dynamic_rnn(self.decode_cell, trg_embed, trg_len, encode_state)

        # 'decode_outputs' is a tensor of shape [batch_size, max_time, cell_state_size], cell_state_size = HIDDEN_SIZE
        output = tf.reshape(decode_outputs, [-1, HIDDEN_SIZE])
        logits = tf.matmul(output, self.softmax_weights) + self.softmax_biases
        cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(trg_labels, [-1]), logits=logits)

        trg_label_weights = tf.sequence_mask(trg_len, maxlen=tf.shape(trg_labels)[1], dtype=tf.float32)
        trg_label_weights = tf.reshape(trg_label_weights, [-1])
        cost = tf.reduce_sum(cross_entropy_loss * trg_label_weights)
        cost_per_token = cost / tf.reduce_sum(trg_label_weights)

        batch_size = tf.shape(trg_input)[0]
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(cost / tf.to_float(batch_size),
                             trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

        return cost_per_token, train_op

    def inference(self, src_input):
        # 虽然输入只有一个句子，但因为dynamic_rnn要求输入是batch的形式，因此这里
        # 将输入句子整理为大小为1的batch。
        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embeddings, src_input)

        # 使用dynamic_rnn构造编码器。这一步与训练时相同。
        with tf.variable_scope("encoder"):
            encode_outputs, encode_state = tf.nn.dynamic_rnn(
                self.encode_cell, src_emb, src_size, dtype=tf.float32)

        # 设置解码的最大步数。这是为了避免在极端情况出现无限循环的问题。
        MAX_DEC_LEN = 100

        with tf.variable_scope("decoder/rnn/multi_rnn_cell"):
            # 使用一个变长的TensorArray来存储生成的句子。
            init_array = tf.TensorArray(dtype=tf.int32, size=0,
                                        dynamic_size=True, clear_after_read=False)
            # 填入第一个单词<sos>作为解码器的输入。
            init_array = init_array.write(0, config.SOS_IDX)
            # 构建初始的循环状态。循环状态包含循环神经网络的隐藏状态，保存生成句子的
            # TensorArray，以及记录解码步数的一个整数step。
            init_loop_var = (encode_state, init_array, 0)

            # tf.while_loop的循环条件：
            # 循环直到解码器输出<eos>，或者达到最大步数为止。
            def continue_loop_condition(state, trg_ids, step):
                return tf.reduce_all(tf.logical_and(
                    tf.not_equal(trg_ids.read(step), config.EOS_IDX),
                    tf.less(step, MAX_DEC_LEN - 1)))

            def loop_body(state, trg_ids, step):
                # 读取最后一步输出的单词，并读取其词向量。
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embeddings,
                                                 trg_input)
                # 这里不使用dynamic_rnn，而是直接调用dec_cell向前计算一步。
                decode_outputs, next_state = self.decode_cell.call(
                    state=state, inputs=trg_emb)
                # 计算每个可能的输出单词对应的logit，并选取logit值最大的单词作为
                # 这一步的而输出。
                output = tf.reshape(decode_outputs, [-1, HIDDEN_SIZE])
                logits = (tf.matmul(output, self.softmax_weights)
                          + self.softmax_biases)
                next_id = tf.argmax(logits, axis=1, output_type=tf.int32)
                # 将这一步输出的单词写入循环状态的trg_ids中。
                trg_ids = trg_ids.write(step+1, next_id[0])
                return next_state, trg_ids, step+1

            # 执行tf.while_loop，返回最终状态。
            state, trg_ids, step = tf.while_loop(
                continue_loop_condition, loop_body, init_loop_var)
            return trg_ids.stack()


CKPT_PATH = "ckpt/seq2seq"


def run_epoch(session, cost_op, train_op, saver, step):
    while True:
        try:
            cost, _ = session.run([cost_op, train_op])
            if step % 100 == 0:
                print(current_time(), "step: %d, per token cost is %.3f" % (step, cost))
            if step % 2000 == 0:
                saver.save(session, CKPT_PATH, global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step


NUM_EPOCH = 8

cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
cfg.gpu_options.allow_growth = True


def train(model):
    with tf.device('/cpu:0'):
        data = create_src_trg_dataset(config.TRAIN_SRC_PATH, config.TRAIN_TRG_PATH)
        iterator = data.make_initializable_iterator()
        (src_input, src_size), (trg_input, trg_size, trg_labels) = iterator.get_next()

    cost_op, train_op = model.forward(src_input, src_size, trg_input, trg_size, trg_labels)

    saver = tf.train.Saver()

    print(current_time(), "training starts...")

    step = 0
    with tf.Session(config=cfg) as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            print(current_time(), "epoch: %d" % (i + 1))
            sess.run(iterator.initializer)
            step = run_epoch(sess, cost_op, train_op, saver, step)

    print(current_time(), "training finishes...")


def test(model):
    # 定义个测试句子。
    test_en_text = "This is a test . <eos>"
    print(test_en_text)

    # 根据英文词汇表，将测试句子转为单词ID。
    with codecs.open(config.SRC_VOCAB_FILE, "r", "utf-8") as f_vocab:
        src_vocab = [w.strip() for w in f_vocab.readlines()]
        src_id_dict = dict((src_vocab[x], x) for x in range(len(src_vocab)))
    test_en_ids = [(src_id_dict[token] if token in src_id_dict else src_id_dict['<unk>'])
                   for token in test_en_text.split()]
    print(test_en_ids)

    # 建立解码所需的计算图。
    output_op = model.inference(test_en_ids)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, "%s-14000" % CKPT_PATH) # step number maybe changed

    # 读取翻译结果。
    output_ids = sess.run(output_op)
    print(output_ids)

    # 根据中文词汇表，将翻译结果转换为中文文字。
    with codecs.open(config.TRG_VOCAB_FILE, "r", "utf-8") as f_vocab:
        trg_vocab = [w.strip() for w in f_vocab.readlines()]
    output_text = ''.join([trg_vocab[x] for x in output_ids])

    # 输出翻译结果。
    print(output_text.encode('utf8').decode(sys.stdout.encoding))
    sess.close()


def main():
    with tf.variable_scope("seq2seq_model", reuse=None,
                           initializer=tf.random_normal_initializer(-0.05, 0.05)):
        model = Seq2seqModel()

    # train
    train(model)

    # test
    test(model)


if __name__ == "__main__":
    main()
