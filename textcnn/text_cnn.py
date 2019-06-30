import tensorflow as tf
import config


class CNNModel(object):
    '''
    def __init__(self):
        with tf.device('/cpu:0'):
            self.dropout_keep_prob_ph = tf.placeholder(tf.float32, name="dropout_keep_prob")
    '''

    def forward(self, content, dropout_keep_prob_ph):
        with tf.device('/cpu:0'):
            embeddings = tf.get_variable('embeddings', [config.VOCAB_SIZE, config.EMBED_SIZE], dtype=tf.float32,
                                         initializer=tf.random_normal_initializer(-1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, content, name='embedding_lookup')

        pooled_outputs = []
        for i, conv_filter_kernel_size in enumerate(config.CONV_FILTER_KERNEL_SIZES):
            with tf.variable_scope('cnn-%d' % conv_filter_kernel_size):
                # conv layer
                conv = tf.layers.conv1d(embed,
                                        config.CONV_FILTER_NUM,
                                        conv_filter_kernel_size,
                                        activation=tf.nn.relu,
                                        name='conv-%d' % conv_filter_kernel_size)
                # global max pooling layer
                gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp-%d' % conv_filter_kernel_size)
                pooled_outputs.append(gmp)
        h_pool = tf.concat(pooled_outputs, 1)

        with tf.variable_scope('fc1'):
            fc = tf.layers.dense(h_pool, config.HIDDEN_SIZE, name='h_pool')
            fc = tf.contrib.layers.dropout(fc, dropout_keep_prob_ph)
            fc = tf.nn.relu(fc)

        with tf.variable_scope('fc2'):
            logits = tf.layers.dense(fc, config.NUM_CLASSES, name='logits')

        return logits

    def predict(self, logits, label):
        with tf.name_scope("output"):
            # prediction
            preds = tf.argmax(tf.nn.softmax(logits), 1, name='predictions')
            # accuracy
            correct_preds = tf.equal(tf.argmax(label, 1), preds)
            acc = tf.reduce_mean(tf.cast(correct_preds, tf.float32), name='accuracy')

        return preds, acc

    def opt(self, logits, label):
        with tf.name_scope("loss"):
            # loss
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
            train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss_op)

        return loss_op, train_op
