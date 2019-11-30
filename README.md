# UBiSE: Unsupervised Bilingual Sentiment Embeddings
Code for [Learning Bilingual Sentiment-specific word Embeddings without Cross-ligual supervision](https://www.aclweb.org/anthology/N19-1040.pdf) (NAACL-HLT 2019)

## Citation
```
@inproceedings{feng2019learning,
  title={Learning Bilingual Sentiment-Specific Word Embeddings without Cross-lingual Supervision},
  author={Feng, Yanlin and Wan, Xiaojun},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  pages={420--429},
  year={2019}
}
```

## Usage

**attention_classify.py**

Script for learning monolingual sentiment-specific vectors.

**ubise.py**

The implementation of the self-learning algorithm. It takes the monolingual sentimental vectors input and outputs two projection matrices.

**dan_eval.py**

Train a DAN to evaluate the projection matrices on cross-lingual sentiment analysis.
