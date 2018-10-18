import numpy as np
import argparse
import pickle
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from itertools import accumulate
from utils.dataset import *
from utils.math import *
from utils.bdi import *
from utils.model import *


COLORS = ['b', 'b', 'r', 'g', 'k', 'y', 'c']
EMB_PATH = 'pickle/%s.bin'
SENTI_PATH = 'categories/categories.%s'


def load_W_source(model_path):
    with tf.variable_scope('projection', reuse=tf.AUTO_REUSE):
        W_source = tf.get_variable(
            'W_source', dtype=tf.float32, initializer=tf.constant(np.zeros((300, 300), dtype=np.float32)))

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, model_path)
        W_source_ = sess.run(W_source)

    return W_source_


def main(args):
    for infile in args.W:
        dic = load_model(infile)
        W_src = dic['W_source']
        W_trg = dic['W_target']
        src_lang = dic['source_lang']
        trg_lang = dic['target_lang']
        model = dic['model']
        with open(EMB_PATH % src_lang, 'rb') as fin:
            src_wv = pickle.load(fin)
        with open(EMB_PATH % trg_lang, 'rb') as fin:
            trg_wv = pickle.load(fin)
        src_senti_words = SentiWordSet(SENTI_PATH % src_lang).to_index(src_wv)
        trg_senti_words = SentiWordSet(SENTI_PATH % trg_lang).to_index(trg_wv)
        src_offsets = [0] + list(accumulate([len(t) for t in src_senti_words.wordsets]))
        trg_offsets = [0] + list(accumulate([len(t) for t in trg_senti_words.wordsets]))
        src_word_idx = sum(src_senti_words.wordsets, [])
        trg_word_idx = sum(trg_senti_words.wordsets, [])

        if model == 'ubise':
            src_proj_emb = np.dot(src_wv.embedding, W_src)
            trg_proj_emb = np.dot(trg_wv.embedding, W_trg)
            length_normalize(src_proj_emb, inplace=True)
            length_normalize(trg_proj_emb, inplace=True)
        elif model == 'ubi':
            src_proj_emb = np.dot(src_wv.embedding, W_src)
            trg_proj_emb = np.dot(trg_wv.embedding, W_trg)

        fig, ax = plt.subplots()

        X = src_proj_emb[src_word_idx]
        ax = fig.add_subplot(121)
        X = TSNE(2, verbose=2).fit_transform(X)
        for i, label in enumerate(src_senti_words.labels):
            tmp = X[src_offsets[i]:src_offsets[i + 1]]
            ax.scatter(tmp[:, 0], tmp[:, 1], s=10, label=label, color=COLORS[i])
        ax.legend()
        ax.set_title(infile + '-source')

        X = trg_proj_emb[trg_word_idx]
        ax = fig.add_subplot(122)
        X = TSNE(2, verbose=2).fit_transform(X)
        for i, label in enumerate(trg_senti_words.labels):
            tmp = X[trg_offsets[i]:trg_offsets[i + 1]]
            ax.scatter(tmp[:, 0], tmp[:, 1], s=10, label=label, color=COLORS[i])
        ax.legend()
        ax.set_title(infile + '-target')

        fig.set_size_inches(20, 8)

        outfile = 'result/' + infile.split('/')[-1].replace('.bin', '') + '.png'
        fig.savefig(outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('W',
                        nargs='+',
                        help='W')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='random seed')

    args = parser.parse_args()
    np.random.seed(args.seed)
    main(args)
