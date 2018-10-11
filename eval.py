import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
import argparse
import pickle
import warnings
import logging
from multiprocessing import cpu_count
from utils import utils
from utils.cupy_utils import *

@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=UndefinedMetricWarning)
def main(args):
    logging.info(str(args))

    src_wv = utils.WordVecs(args.source_embedding, emb_format=args.format).normalize(args.normalize)
    trg_wv = utils.WordVecs(args.target_embedding, emb_format=args.format).normalize(args.normalize)
    src_proj_emb = np.empty(src_wv.embedding.shape, dtype=np.float32)
    trg_proj_emb = np.empty(trg_wv.embedding.shape, dtype=np.float32)

    for infile in args.W:
        with open(infile, 'rb') as fin:
            W_src, W_trg = pickle.load(fin)
        src_wv.embedding.dot(W_src, out=src_proj_emb)
        utils.length_normalize(src_proj_emb, inplace=True)
        trg_wv.embedding.dot(W_trg, out=trg_proj_emb)
        utils.length_normalize(trg_proj_emb, inplace=True)
        src_ds = utils.SentimentDataset(args.source_dataset).to_index(src_wv, binary=args.binary).to_vecs(src_proj_emb, shuffle=True)
        trg_ds = utils.SentimentDataset(args.target_dataset).to_index(trg_wv, binary=args.binary).to_vecs(trg_proj_emb, shuffle=True)
        train_x = np.concatenate((src_ds.train[0], src_ds.dev[0], src_ds.test[0]), axis=0)
        train_y = np.concatenate((src_ds.train[1], src_ds.dev[1], src_ds.test[1]), axis=0)
        test_x = trg_ds.train[0]
        test_y = trg_ds.train[1]

        param_grid = {
            'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000],
        }
        svc = svm.LinearSVC()
        clf = GridSearchCV(svc, param_grid, scoring='f1_macro', n_jobs=cpu_count())
        clf.fit(train_x, train_y)
        print('------------------------------------------------------')
        print('Result for {0}:'.format(infile))
        print('Test F1_macro: {0:.4f}'.format(clf.score(test_x, test_y)))
        print('Best params: {0}'.format(clf.best_params_))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('W',
                        nargs='+',
                        default=['checkpoints/senti.bin'],
                        help='W_src and W_trg')
    parser.add_argument('--format',
                        choices=['word2vec_binary', 'fasttext_text'],
                        default='fasttext_text',
                        help='word embedding format')
    parser.add_argument('-bi', '--binary',
                        action='store_true',
                        help='use 2-class set up')
    parser.add_argument('-se', '--source_embedding',
                        default='./emb/en.bin',
                        help='monolingual word embedding of the source language (default: ./emb/en.bin)')
    parser.add_argument('-te', '--target_embedding',
                        default='./emb/es.bin',
                        help='monolingual word embedding of the target language (default: ./emb/es.bin)')
    parser.add_argument('-sd', '--source_dataset',
                        default='./datasets/en/opener_sents/',
                        help='source sentiment dataset')
    parser.add_argument('-td', '--target_dataset',
                        default='./datasets/es/opener_sents/',
                        help='target sentiment dataset')
    parser.add_argument('--normalize',
                        choices=['unit', 'center'],
                        nargs='*',
                        default=['center', 'unit'],
                        help='normalization actions')
    parser.add_argument('--debug',
                        action='store_const',
                        dest='loglevel',
                        default=logging.INFO,
                        const=logging.DEBUG,
                        help='print debug info')

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel, format='%(asctime)s: %(message)s')

    main(args)
