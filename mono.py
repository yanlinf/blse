import numpy as np
import logging
import argparse
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from multiprocessing import cpu_count
from utils.dataset import *
from utils.math import *
from utils.bdi import *


def lookup_and_shuffle(X, y, emb, binary=False):
    X_new = np.zeros((len(X), emb.shape[1]))
    for i, line in enumerate(X):
        if len(line) == 0:
            logging.warning('ZERO LENGTH EXAMPLE')
            continue
        X_new[i] = np.mean(emb[line], axis=0)
    X = X_new

    perm = np.random.permutation(X.shape[0])
    X, y = X[perm], y[perm]
    if binary:
        y = (y >= 2).astype(np.int32)
    return X, y


def main(args):
    logging.info(str(args))

    # load word embedding
    target_wordvec = WordVecs(args.target_embedding, normalize=args.normalize)

    target_dataset = SentimentDataset(args.target_dataset).to_index(target_wordvec)

    # embedding lookup and prepare traning data

    train_x, train_y = lookup_and_shuffle(*target_dataset.train, target_wordvec.embedding, args.binary)
    test_x, test_y = lookup_and_shuffle(*target_dataset.test, target_wordvec.embedding, args.binary)

    # train linear SVM classifier and tune parameter C
    param_grid = {
        'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
    }
    svc = svm.LinearSVC()
    clf = GridSearchCV(svc, param_grid, scoring='f1_macro', n_jobs=cpu_count())

    clf.fit(train_x, train_y)

    print('Test F1_macro: %.4f' % clf.score(test_x, test_y))
    print('Best params: ', clf.best_params_)
    print('CV result:', clf.cv_results_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-bi', '--binary',
                        help='use 2-class set up',
                        action='store_true')
    parser.add_argument('-tl', '--target_lang',
                        help='target language: en/es/ca/eu (default: es)',
                        default='es')
    parser.add_argument('-te', '--target_embedding',
                        help='monolingual word embedding of the target language (default: ./emb/es.bin)',
                        default='./emb/es.bin')
    parser.add_argument('-td', '--target_dataset',
                        help='sentiment dataset of the target language',
                        default='./datasets/es/opener_sents/')
    parser.add_argument('--normalize',
                        help='mean center and normalize word vectors',
                        action='store_true')
    parser.add_argument('--debug',
                        help='print debug info',
                        action='store_const',
                        dest='loglevel',
                        default=logging.INFO,
                        const=logging.DEBUG)

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel,
                        format='%(asctime)s: %(levelname)s: %(message)s')
    main(args)
