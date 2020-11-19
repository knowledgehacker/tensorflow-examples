# -*- coding: utf-8 -*-

import tensorflow.compat.v1 as tf

import config
from input_feed import init_next_batch
from utils import current_time


cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
cfg.gpu_options.allow_growth = True


def test():
    print(current_time(), "testing starts...")

    g = tf.Graph()
    with tf.Session(graph=g, config=cfg) as sess:
        # load trained model
        load_model(sess, config.CKPT_DIR)

        # create iterator for test dataset
        dataset_init_op, get_next_op = init_next_batch(config.TEST_PATH, test=True)

        # important!!! Don't call 'tf.global_variables_initializer().run()' when doing inference using trained model
        #tf.global_variables_initializer().run()
        sess.run(dataset_init_op)

        # get prediction and other dependent tensors from the graph in the trained model for inference
        droupout_keep_prob_ph = g.get_tensor_by_name("dropout_keep_prob:0")
        preds_op = g.get_tensor_by_name("output/predictions:0")
        acc_op = g.get_tensor_by_name("output/accuracy:0")

        try:
            while True:
                # get next batch from test dataset iterator, feed to 'get_next' operator in the graph loaded from trained model
                content, len, label = sess.run(get_next_op)

                preds, acc = sess.run([preds_op, acc_op], feed_dict={'get_next:0': content, 'get_next:1': len, 'get_next:2': label,
                                                           droupout_keep_prob_ph: config.TEST_KEEP_PROB})
                print('accuracy: %.3f' % acc)
        except tf.errors.OutOfRangeError:
            pass

    print(current_time(), "testing finishes...")


def load_model(sess, ckpt_dir):
    ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
    print("ckpt_file: %s" % ckpt_file)
    saver = tf.train.import_meta_graph("{}.meta".format(ckpt_file))
    saver.restore(sess, ckpt_file)


def main():
    # test
    test()


if __name__ == "__main__":
    main()
