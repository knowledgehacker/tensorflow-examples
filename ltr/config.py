TRAIN_DATA_PATH = "data/set1.train.txt" 
TEST_DATA_PATH = "data/set1.test.txt"

FEATURE_NUM = 700
MAX_DOC_NUM = 100

# Parameters to the scoring function.
BATCH_SIZE = 32
SHUFFLE_SIZE = 1000
HIDDEN_LAYER_DIMS = ["256", "128", "64"]

LOSS_FUNC = "pairwise_logistic_loss"

EPOCH_NUM = 10

CKPT_DIR = 'ckpt'