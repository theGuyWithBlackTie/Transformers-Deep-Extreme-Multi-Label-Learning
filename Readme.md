This repository is the implementation of the SOTA paper "Deep Extreme Multi-Label Learning" (DXML) by Wenjie Zhang, Junchi Yan, Xiangfeng Wang and Hongyan Zha. ArXiv Link: (https://arxiv.org/abs/1704.03718)

Original paper idea doesn't include BERT and hence expects the input text to be either converted in BoW format or to use external embeddings like Word2Vec or Glove Embeddings. The datasets used in the paper are present in BoW formats.

This repository uses BERT transformer and its BertTokenizer instead of BoW or external embeddings. The dataset used is EURLEX dataset from HuggingFace (https://huggingface.co/datasets/eurlex)

The results obtained are:
P@1: 0.66 <br>
P@3: 0.56 <br>
P@5: 0.47

With BERT, the model achieved this score within 4 epochs and almost 4hrs of whole training time. Same performace is achieved in BoW case in 70 epochs but each epoch takes 10 seconds.

The code for BoW format can be found here: https://github.com/theGuyWithBlackTie/Deep-Extreme-Multi-Label-Learning