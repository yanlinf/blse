import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer, f1_score
import argparse
from utils import utils
from multiprocessing import cpu_count
import pickle
import logging


def get_W_target(source_emb, target_emb, dict_obj, orthogonal):
    """
    source_emb: np.ndarray of shape (source_vocab_size, vec_dim)
    target_emb: np.ndarray of shape (target_vocab_size, vec_dim)
    dict_obj: np.ndarray of shape (dict_size, 2)
    orthogonal: bool

    Returns: np.ndarray of shape (vec_dim, vec_dim)
    """
    X_source = source_emb[dict_obj[:, 0]]  # shape (dict_size, vec_dim)
    X_target = target_emb[dict_obj[:, 1]]  # shape (dict_size, vec_dim)

    if orthogonal:
        u, s, vt = np.linalg.svd(np.dot(X_source.T, X_target))
        W_target = np.dot(vt.T, u.T)
    else:
        W_target = np.matmul(np.linalg.pinv(X_target), X_source)

    return W_target


def get_W_source(source_emb, target_emb, dict_obj, orthogonal):
    """
    source_emb: np.ndarray of shape (source_vocab_size, vec_dim)
    target_emb: np.ndarray of shape (target_vocab_size, vec_dim)
    dict_obj: np.ndarray of shape (dict_size, 2)
    orthogonal: bool

    Returns: np.ndarray of shape (vec_dim, vec_dim)
    """
    X_source = source_emb[dict_obj[:, 0]]  # shape (dict_size, vec_dim)
    X_target = target_emb[dict_obj[:, 1]]  # shape (dict_size, vec_dim)

    if orthogonal:
        u, s, vt = np.linalg.svd(np.dot(X_target.T, X_source))
        W_source = np.dot(vt.T, u.T)
    else:
        W_source = np.matmul(np.linalg.pinv(X_source), X_target)

    return W_source


def cal_proj_loss(source_emb, target_emb, dict_obj):
    X_source = source_emb[dict_obj[:, 0]]  # shape (dict_size, vec_dim)
    X_target = target_emb[dict_obj[:, 1]]  # shape (dict_size, vec_dim)
    return np.sum(np.square(X_source - X_target))


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
    source_wordvecs = utils.WordVecs(
        args.source_embedding, normalize=args.normalize)
    target_wordvecs = utils.WordVecs(
        args.target_embedding, normalize=args.normalize)

    # load bilingual lexicon
    dict_obj = utils.BilingualDict(args.dictionary).filter(
        lambda x: x[0] != '-').get_indexed_dictionary(source_wordvecs, target_wordvecs)

    source_dataset = utils.SentimentDataset(
        args.source_dataset).to_index(source_wordvecs)
    target_dataset = utils.SentimentDataset(
        args.target_dataset).to_index(target_wordvecs)

    # create bilingual embedding
    if args.project_source:
        proj_source_emb = np.matmul(source_wordvecs.embedding, get_W_source(
            source_wordvecs.embedding, target_wordvecs.embedding, dict_obj, args.orthogonal))
        proj_target_emb = target_wordvecs.embedding
    else:
        proj_source_emb = source_wordvecs.embedding
        proj_target_emb = np.matmul(target_wordvecs.embedding, get_W_target(
            source_wordvecs.embedding, target_wordvecs.embedding, dict_obj, args.orthogonal))

    logging.info('projection loss before projection: %.2f' % cal_proj_loss(
        source_wordvecs.embedding, target_wordvecs.embedding, dict_obj))
    logging.info('projection loss after projection: %.2f' % cal_proj_loss(
        proj_source_emb, proj_target_emb, dict_obj))

    # embedding lookup and prepare traning data
    train_x = source_dataset.train[
        0] + source_dataset.dev[0] + source_dataset.test[0]
    train_y = np.concatenate(
        (source_dataset.train[1], source_dataset.dev[1], source_dataset.test[1]), axis=0)

    train_x, train_y = lookup_and_shuffle(
        train_x, train_y, proj_source_emb, args.binary)
    test_x, test_y = lookup_and_shuffle(*target_dataset.train, proj_target_emb, args.binary)

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
    parser.add_argument('-sl', '--source_lang',
                        help='source language: en/es/ca/eu (default: en)',
                        default='eu')
    parser.add_argument('-tl', '--target_lang',
                        help='target language: en/es/ca/eu (default: es)',
                        default='es')
    parser.add_argument('-se', '--source_embedding',
                        help='monolingual word embedding of the source language (default: ./emb/en.bin)',
                        default='./emb/en.bin')
    parser.add_argument('-te', '--target_embedding',
                        help='monolingual word embedding of the target language (default: ./emb/es.bin)',
                        default='./emb/es.bin')
    parser.add_argument('-d', '--dictionary',
                        help='bilingual dictionary of source and target language (default: ./lexicons/bingliu/en-es.txt',
                        default='./lexicons/bingliu/en-es.txt')
    parser.add_argument('-sd', '--source_dataset',
                        help='sentiment dataset of the source language',
                        default='./datasets/en/opener_sents/')
    parser.add_argument('-td', '--target_dataset',
                        help='sentiment dataset of the target language',
                        default='./datasets/es/opener_sents/')
    parser.add_argument('--project_source',
                        help='project source embedding (default: project target)',
                        action='store_true')
    parser.add_argument('--normalize',
                        help='mean center and normalize word vectors',
                        action='store_true')
    parser.add_argument('--orthogonal',
                        help='apply orthogonal restriction to the projection matrix',
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
