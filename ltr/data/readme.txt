1. download "yahoo ltr challenge dataset"
https://webscope.sandbox.yahoo.com, DATASETS -> Competition Data -> C14 - Yahoo! Learning to Rank Challenge (421 MB)
Uncompress it and copy set1.train.txt and set1.test.txt here.

2. data preprocess
Tensorflow Ranking requires input to be represented in LIBSVM format as follows：
<relevance int> qid:<query_id int> [<feature_id int>:<feature_value float>]
yahoo ltr challenge dataset is preprocessed already, and data format conforms to Tensorflow Ranking requirement, so you can use it directly.
If you use other dataset, please preprocess it to conform to Tenforflow Ranking requirement if necessary.
