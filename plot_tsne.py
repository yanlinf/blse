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


COLORS = ['b', 'r', 'g', 'k', 'y', 'c']
EMB_PATH = 'pickle/%s.bin'
SENTI_PATH = 'categories/categories.%s'


def main(args):
    fig = plt.figure()

    assert len(args.W) == 3
    for j, infile in enumerate(args.W):
        np.random.seed(args.seed)

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
        elif model == 'blse':
            src_proj_emb = np.dot(src_wv.embedding, W_src)
            trg_proj_emb = np.dot(trg_wv.embedding, W_trg)
        else:
            src_proj_emb = np.dot(src_wv.embedding, W_src)
            trg_proj_emb = np.dot(trg_wv.embedding, W_trg)

        ax = fig.add_subplot('13{}'.format(j + 1))
        X = trg_proj_emb[trg_word_idx]
        # X = X[:, :2]
        X = TSNE(2, verbose=2).fit_transform(X)
        for i, label in enumerate(trg_senti_words.labels):
            tmp = X[trg_offsets[i]:trg_offsets[i + 1]]
            ax.scatter(tmp[:, 0], tmp[:, 1], s=10, label=label, color=COLORS[i])
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        # ax.legend()
        if j == 1:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, shadow=True, ncol=6, prop={'size': 22})

    fig.subplots_adjust(wspace=0.06)
    fig.set_size_inches(24, 7)
    fig.savefig('result/visual.pdf', format='pdf', bbox_inches='tight')


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
    main(args)
