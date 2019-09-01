import tensorflow as tf
import tensorflow_ranking as tfr
from config import TRAIN_DATA_PATH, TEST_DATA_PATH, FEATURE_NUM, HIDDEN_LAYER_DIMS, LOSS_FUNC, EPOCH_NUM, CKPT_DIR
from input_feed import create_dataset

CKPT_PATH = '%s/ltr' % CKPT_DIR


def example_feature_columns():
    """Returns the example feature columns."""
    feature_names = [
        "%d" % (i + 1) for i in range(0, FEATURE_NUM)
    ]
    return {
        name: tf.feature_column.numeric_column(
            name, shape=(1,), default_value=0.0) for name in feature_names
    }


def make_score_fn():
    """Returns a scoring function to build `EstimatorSpec`."""

    def _score_fn(context_features, group_features, mode, params, config):
        """Defines the network to score a documents."""
        del params
        del config
        # Define input layer.
        example_input = [
            tf.layers.flatten(group_features[name])
            for name in sorted(example_feature_columns())
        ]
        input_layer = tf.concat(example_input, 1)

        cur_layer = input_layer
        for i, layer_dim in enumerate(int(d) for d in HIDDEN_LAYER_DIMS):
            cur_layer = tf.layers.dense(
                cur_layer,
                units=layer_dim,
                activation="tanh")
        logits = tf.layers.dense(cur_layer, units=1)
        return logits

    return _score_fn


def eval_metric_fns(topn_list):
    metric_fns = {}
    metric_fns.update({
        "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
            tfr.metrics.RankingMetricKey.NDCG, topn=topn)
        for topn in topn_list
    })

    return metric_fns


def get_estimator(hparams):
    """Create a ranking estimator.

    Args:
        hparams: (tf.contrib.training.HParams) a hyperparameters object.

    Returns:
        tf.learn `Estimator`.
    """
    def _train_op_fn(loss):
        """Defines train op used in ranking head."""
        return tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=hparams.learning_rate,
            optimizer="Adagrad")

    ranking_head = tfr.head.create_ranking_head(
        loss_fn=tfr.losses.make_loss_fn(LOSS_FUNC),
        eval_metric_fns=eval_metric_fns([5, 10, 20]),
        train_op_fn=_train_op_fn)

    return tf.estimator.Estimator(
        model_fn=tfr.model.make_groupwise_ranking_fn(
            group_score_fn=make_score_fn(),
            group_size=1,
            transform_fn=None,
            ranking_head=ranking_head),
        model_dir = CKPT_PATH,
        params=hparams)


cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
cfg.gpu_options.allow_growth = True


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    ranker = get_estimator(tf.contrib.training.HParams(learning_rate=0.1))

    # train
    ranker.train(input_fn=lambda: create_dataset(TRAIN_DATA_PATH, False), steps=3000)

    # test
    ranker.evaluate(input_fn=lambda: create_dataset(TEST_DATA_PATH, True), steps=1000)


if __name__ == "__main__":
    main()