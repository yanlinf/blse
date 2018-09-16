import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.manifold import TSNE
from utils import utils
from itertools import accumulate


def load_W_source(model_path):
    with tf.variable_scope('projection', reuse=tf.AUTO_REUSE):
        W_source = tf.get_variable(
            'W_source', dtype=tf.float32, initializer=tf.constant(np.zeros((300, 300), dtype=np.float32)))

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, model_path)
        W_source_ = sess.run(W_source)

    return W_source_


def main(args):
    wordvecs = utils.WordVecs(args.source_embedding)
    senti_words = utils.SentiWordSet(args.senti_words).to_index(wordvecs)
    offsets = [0] + list(accumulate([len(t) for t in senti_words.wordsets]))
    words = sum(senti_words.wordsets, [])

    X = wordvecs.embedding[words]

    fig, ax = plt.subplots()
    X = TSNE(2, verbose=2).fit_transform(X)
    for i, label in enumerate(senti_words.labels):
        tmp = X[offsets[i]:offsets[i + 1]]
        ax.scatter(tmp[:, 0], tmp[:, 1], s=10, label=label)
    ax.legend()

    # fig, ax = plt.subplots()
    # X_proj = np.matmul(X, W_target)
    # X_proj = TSNE(2, verbose=2).fit_transform(X_proj)
    # for i, label in enumerate(senti_words.labels):
    #     tmp = X_proj[offsets[i]:offsets[i + 1]]
    #     ax.scatter(tmp[:, 0], tmp[:, 1], s=10, label=label)
    # ax.legend()

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-se', '--source_embedding',
                        type=str,
                        default='./emb/en.bin',
                        help='word embeddings (in Word2Vec binary format)')
    parser.add_argument('-sw', '--senti_words',
                        type=str,
                        default='./categories/categories.en',
                        help='sentiment words')
    args = parser.parse_args()
    main(args)
