import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import f1_score
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
import argparse
import pickle
import warnings
import logging
from multiprocessing import cpu_count
from utils.dataset import *
from utils.math import *
from utils.bdi import *
from utils.cupy_utils import *
from utils.model import *


@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=UndefinedMetricWarning)
def main(args):
    print(str(args))

    if args.output is not None:
        with open(args.output, 'w', encoding='utf-8') as fout:
            fout.write('infile,src_lang,trg_lang,model,is_binary,f1_macro,best_C\n')
    for infile in args.W:
        dic = load_model(infile)
        W_src = dic['W_source']
        W_trg = dic['W_target']
        src_lang = dic['source_lang']
        trg_lang = dic['target_lang']
        model = dic['model']
        with open('pickle/%s.bin' % src_lang, 'rb') as fin:
            src_wv = pickle.load(fin)
        with open('pickle/%s.bin' % trg_lang, 'rb') as fin:
            trg_wv = pickle.load(fin)
        src_proj_emb = np.empty(src_wv.embedding.shape, dtype=np.float32)
        trg_proj_emb = np.empty(trg_wv.embedding.shape, dtype=np.float32)
        if model == 'ubise':
            src_wv.embedding.dot(W_src, out=src_proj_emb)
            length_normalize(src_proj_emb, inplace=True)
            trg_wv.embedding.dot(W_trg, out=trg_proj_emb)
            length_normalize(trg_proj_emb, inplace=True)
        elif model == 'ubi':
            src_wv.embedding.dot(W_src, out=src_proj_emb)
            trg_wv.embedding.dot(W_trg, out=trg_proj_emb)
        elif model == 'blse':
            src_wv.embedding.dot(W_src, out=src_proj_emb)
            trg_wv.embedding.dot(W_trg, out=trg_proj_emb)

        for is_binary in (True, False):
            src_ds = SentimentDataset('datasets/%s/opener_sents/' % src_lang).to_index(src_wv, binary=is_binary).to_vecs(src_proj_emb, shuffle=True)
            trg_ds = SentimentDataset('datasets/%s/opener_sents/' % trg_lang).to_index(trg_wv, binary=is_binary).to_vecs(trg_proj_emb, shuffle=True)
            train_dev_x = np.concatenate((src_ds.train[0], trg_ds.dev[0]), axis=0)
            train_dev_y = np.concatenate((src_ds.train[1], trg_ds.dev[1]), axis=0)
            train_x = src_ds.train[0]
            train_y = src_ds.train[1]
            test_x = trg_ds.test[0]
            test_y = trg_ds.test[1]

            if args.C is not None:
                clf = svm.LinearSVC(C=args.C)
                clf.fit(train_x, train_y)
                best_C = args.C
                test_score = f1_score(test_y, clf.predict(test_x), average='macro')
            else:
                cv_fold = np.zeros(train_dev_x.shape[0], dtype=np.int32)
                cv_fold[:train_x.shape[0]] = -1
                cv_split = PredefinedSplit(cv_fold)
                param_grid = {
                    'C': [0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300],
                }
                svc = svm.LinearSVC()
                clf = GridSearchCV(svc, param_grid, scoring='f1_macro', n_jobs=cpu_count(), cv=cv_split)
                clf.fit(train_dev_x, train_dev_y)
                best_C = clf.best_params_['C']
                pred = svm.LinearSVC(C=best_C).fit(train_x, train_y).predict(test_x)
                test_score = f1_score(test_y, pred, average='macro')
            print('------------------------------------------------------')
            print('Is binary: {0}'.format(is_binary))
            print('Result for {0}:'.format(infile))
            print('Test F1_macro: {0:.4f}'.format(test_score))
            print('Best C: {0}'.format(best_C))
            if args.output is not None:
                with open(args.output, 'a', encoding='utf-8') as fout:
                    fout.write('{0},{1},{2},{3},{4},{5:.4f},{6:.2f}\n'.format(infile, src_lang,
                                                                              trg_lang, model,
                                                                              is_binary, test_score,
                                                                              best_C))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('W',
                        nargs='+',
                        default=['checkpoints/senti.bin'],
                        help='W_src and W_trg')
    parser.add_argument('-C', '--C',
                        type=float,
                        help='train svm with fixed C')
    parser.add_argument('-o', '--output',
                        help='output file')
    parser.add_argument('--debug',
                        action='store_const',
                        dest='loglevel',
                        default=logging.INFO,
                        const=logging.DEBUG,
                        help='print debug info')

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel, format='%(asctime)s: %(message)s')

    main(args)
