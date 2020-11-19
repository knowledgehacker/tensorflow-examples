import tensorflow.compat.v1 as tf

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
                                         initializer=tf.truncated_normal_initializer(stddev=0.01))
            embed = tf.nn.embedding_lookup(embeddings, content, name='embedding_lookup')

        pooled_outputs = []
        for conv_filter_kernel_size in config.CONV_FILTER_KERNEL_SIZES:
            with tf.variable_scope('cnn-%d' % conv_filter_kernel_size):
                # conv layer, relu activation ensures elements of output > 0
                conv = tf.layers.conv1d(embed,
                                        config.CONV_FILTER_NUM,
                                        conv_filter_kernel_size,
                                        activation=tf.nn.relu,
                                        name='conv-%d' % conv_filter_kernel_size)
                # global max pooling layer, max over a convolution filter
                gmp = tf.reduce_max(conv, axis=1, name='gmp-%d' % conv_filter_kernel_size)
                pooled_outputs.append(gmp)

        # dimension of h_pool: [config.BATCH_SIZE, CONV_FILTER_NUM * CONV_FILTER_KERNEL_SIZES]
        h_pool = tf.concat(pooled_outputs, 1)
        h_dropout = tf.nn.dropout(h_pool, rate=1.0 - dropout_keep_prob_ph)

        with tf.variable_scope('fc1'):
            hl = tf.layers.dense(h_dropout, config.HIDDEN_SIZE, name='hl')
            hl = tf.nn.relu(hl)

        with tf.variable_scope('fc2'):
            logits = tf.layers.dense(hl, config.NUM_CLASSES, name='logits')

        """
        with tf.variable_scope('fc'):
            logits = tf.layers.dense(h_dropout, config.NUM_CLASSES, name='logits')
        """

        return logits

    def predict(self, logits, label):
        with tf.name_scope("output"):
            # prediction
            #preds = tf.argmax(tf.nn.softmax(logits), 1, name='predictions')
            preds = tf.argmax(logits, 1, name='predictions')
            # accuracy
            correct_preds = tf.equal(tf.argmax(label, 1), preds)
            acc = tf.reduce_mean(tf.cast(correct_preds, tf.float32), name='accuracy')

        return preds, acc

    def opt(self, logits, label):
        with tf.name_scope("loss"):
            # loss
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
            train_op = tf.train.AdamOptimizer(learning_rate=config.LEARNING_RATE).minimize(loss_op)

        return loss_op, train_op
