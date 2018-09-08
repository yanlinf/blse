import tensorflow as tf
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import argparse
from utils import utils
from multiprocessing import cpu_count
import pickle
import logging


def get_W_target(source_emb, target_emb, dict_obj):
    """
    source_emb: np.ndarray of shape (source_vocab_size, vec_dim)
    target_emb: np.ndarray of shape (target_vocab_size, vec_dim)
    dict_obj: np.ndarray of shape (dict_size, 2)

    Returns: np.ndarray of shape (vec_dim, vec_dim)
    """
    X_source = source_emb[dict_obj[:, 0]]  # shape (dict_size, vec_dim)
    X_target = target_emb[dict_obj[:, 1]]  # shape (dict_size, vec_dim)
    W_target = np.matmul(np.linalg.pinv(X_target), X_source)
    return W_target


def main(args):
    logging.info(str(args))

    if args.load:
        with open('./tmp/pickled_%dclass.bin' % (2 if args.binary else 4), 'rb') as fin:
            source_wordvec, target_wordvec, dict_obj, train_x, train_y, dev_x, dev_y, test_x, test_y = pickle.load(
                fin)
    else:
        source_wordvec = utils.WordVecs(args.source_embedding)
        target_wordvec = utils.WordVecs(args.target_embedding)

        source_pad_id = source_wordvec.add_word('<PAD>', np.zeros(300))
        target_pad_id = target_wordvec.add_word('<PAD>', np.zeros(300))

        dict_obj = utils.BilingualDict(args.dictionary).filter(
            lambda x: x[0] != '-').get_indexed_dictionary(source_wordvec, target_wordvec)

        source_dataset = utils.SentimentDataset(
            args.source_dataset).to_index(source_wordvec)
        target_dataset = utils.SentimentDataset(
            args.target_dataset).to_index(target_wordvec)

        def pad_and_shuffle(X, y, pad_id, emb, binary=False):
            X = tf.keras.preprocessing.sequence.pad_sequences(
                X, maxlen=64, value=pad_id)
            perm = np.random.permutation(X.shape[0])
            X, y = X[perm], y[perm]
            if binary:
                y = (y >= 2).astype(np.int32)
            X = np.sum(emb[X], axis=1)
            return X, y

        train_x, train_y = pad_and_shuffle(*source_dataset.train, source_pad_id, source_wordvec.embedding, args.binary)
        dev_x, dev_y = pad_and_shuffle(*target_dataset.test, target_pad_id, target_wordvec.embedding, args.binary)
        test_x, test_y = pad_and_shuffle(*target_dataset.train, target_pad_id, target_wordvec.embedding, args.binary)

        # with open('./tmp/pickled_%dclass.bin' % (2 if args.binary else 4), 'wb') as fout:
        #     pickle.dump((source_wordvec, target_wordvec, dict_obj, train_x,
        # train_y, dev_x, dev_y, test_x, test_y), fout, protocol=4)

    W_target = get_W_target(source_wordvec.embedding,
                            target_wordvec.embedding, dict_obj)
    proj_target_emb = np.matmul(target_wordvec.embedding, W_target)

    param_grid = {
        'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30],
    }
    cv_split = PredefinedSplit(np.concatenate(
        (np.full(train_x.shape[0], -1), np.full(dev_x.shape[0], 0)), axis=0))
    svc = svm.LinearSVC()

    clf = GridSearchCV(svc, param_grid, scoring='f1_macro', cv=cv_split, n_jobs=cpu_count(), verbose=3)

    X = np.concatenate((train_x, dev_x), axis=0)
    y = np.concatenate((train_y, dev_y), axis=0)
    clf.fit(X, y)

    print('Test accuracy: %.4f' % clf.score(test_x, test_y))
    print('Best params', clf.best_params_)


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
                        default='./lexicons/bingliu/en-eu.txt')
    parser.add_argument('-sd', '--source_dataset',
                        help='sentiment dataset of the source language',
                        default='./datasets/en/opener_sents/')
    parser.add_argument('-td', '--target_dataset',
                        help='sentiment dataset of the target language',
                        default='./datasets/es/opener_sents/')
    parser.add_argument('-vd', '--vector_dim',
                        help='dimension of each word vector (default: 300)',
                        default=300,
                        type=int)
    parser.add_argument('--load',
                        action='store_true',
                        help='load from pickled object')
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
