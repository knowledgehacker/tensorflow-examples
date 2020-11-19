# -*- coding: utf-8 -*-

import tensorflow.compat.v1 as tf

import config
from input_feed import init_next_batch
from text_cnn import CNNModel
from utils import current_time, restore

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


CKPT_PATH = '%s/textcnn' % config.CKPT_DIR


cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
cfg.gpu_options.allow_growth = True


def train(model):
    print(current_time(), "training starts...")

    #sentence_size_max = restore(config.SENTENCE_FILE)
    sentence_size_max = config.SENTENCE_SIZE_MAX
    print("sentence_size_max=%d" % sentence_size_max)

    g = tf.Graph()
    with g.as_default():
        dropout_keep_prob_ph = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # create iterator for train dataset
        dataset_init_op, get_next_op = init_next_batch(config.TRAIN_PATH, sentence_size_max)
        content, label = get_next_op

        logits = model.forward(content, dropout_keep_prob_ph)
        loss_op, train_op = model.opt(logits, label)
        _, acc_op = model.predict(logits, label)

        saver = tf.train.Saver()

    with tf.Session(graph=g, config=cfg) as sess:

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", loss_op)
        acc_summary = tf.summary.scalar("accuracy", acc_op)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_writer = tf.summary.FileWriter('logs/train_summaries.txt', sess.graph)

        tf.global_variables_initializer().run()

        step = 0
        for i in range(config.NUM_EPOCH):
            print(current_time(), "epoch: %d" % (i + 1))
            sess.run(dataset_init_op)

            while True:
                try:
                    train_summaries, acc, loss, _ = sess.run([train_summary_op, acc_op, loss_op, train_op],
                                                             feed_dict={dropout_keep_prob_ph: config.TRAIN_KEEP_PROB})
                    train_summary_writer.add_summary(train_summaries, step)

                    if step % config.STEPS_PER_CKPT == 0:
                        print(current_time(), "step: %d, loss: %.3f, accuracy: %.3f" % (step, loss, acc))
                    if step % 100 == 0:
                        saver.save(sess, CKPT_PATH, global_step=step)
                    step += 1
                except tf.errors.OutOfRangeError:
                    train_summary_writer.add_summary(train_summaries, step)

                    print(current_time(), "step: %d, loss: %.3f, accuracy: %.3f" % (step, loss, acc))
                    saver.save(sess, CKPT_PATH, global_step=step)
                    break

    print(current_time(), "training finishes...")


def main():
    model = CNNModel()

    # train
    train(model)


if __name__ == "__main__":
    main()
