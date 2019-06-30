# -*- coding: utf-8 -*-

import tensorflow as tf
import config
from input_feed import init_next_batch
from text_cnn import CNNModel
from utils import current_time


CKPT_PATH = '%s/text-classification' % config.CKPT_DIR


def run_epoch(session, dropout_keep_prob_ph, acc_op, loss_op, train_op, saver, step):
    while True:
        try:
            acc, loss, _ = session.run([acc_op, loss_op, train_op], feed_dict={dropout_keep_prob_ph: config.TRAIN_KEEP_PROB})
            if step % 10 == 0:
                print(current_time(), "step: %d, loss: %.3f, accuracy: %.3f" % (step, loss, acc))
            if step % 100 == 0:
                saver.save(session, CKPT_PATH, global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step


cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
cfg.gpu_options.allow_growth = True


def train(model):
    print(current_time(), "training starts...")

    dropout_keep_prob_ph = tf.placeholder(tf.float32, name="dropout_keep_prob")
    # create iterator for train dataset
    dataset_init_op, get_next_op = init_next_batch(config.TRAIN_PATH)
    content, len, label = get_next_op

    logits = model.forward(content, dropout_keep_prob_ph)
    loss_op, train_op = model.opt(logits, label)
    _, acc_op = model.predict(logits, label)

    saver = tf.train.Saver()

    with tf.Session(config=cfg) as sess:
        tf.global_variables_initializer().run()

        step = 0
        for i in range(config.NUM_EPOCH):
            print(current_time(), "epoch: %d" % (i + 1))
            sess.run(dataset_init_op)
            step = run_epoch(sess, dropout_keep_prob_ph, acc_op, loss_op, train_op, saver, step)
        saver.save(sess, CKPT_PATH, global_step=step)

    print(current_time(), "training finishes...")


def main():
    model = CNNModel()

    # train
    train(model)


if __name__ == "__main__":
    main()
