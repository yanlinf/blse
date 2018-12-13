import argparse
import pickle
import logging
import sys
import os
import re
import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
from sklearn.metrics import f1_score
from sklearn import svm
from multiprocessing import cpu_count
from utils.dataset import *
from utils.math import *
from utils.bdi import *
from utils.cupy_utils import *
from utils.model import *


def DEBUG(arg):
    logging.debug('DEBUG %s' % str(arg))


def topkidx(x, k, inplace=False):
    xp = get_array_module(x, k)
    res = xp.empty(k, dtype=xp.int32)
    min_val = x.min()
    if not inplace:
        x = x.copy()
    for i in range(k):
        res[i] = xp.argmax(x)
        x[res[i]] = min_val
    return res


def ubise(P, N, a, W, p):
    xp = get_array_module(P, N, a, W)
    pw = P.dot(W)
    nw = N.dot(W)
    if p == 1.0:
        k1 = P.shape[0]
        k2 = N.shape[0]
        pi = xp.arange(k1)
        ni = xp.arange(k2)
    else:
        k1 = int(P.shape[0] * p)
        k2 = int(N.shape[0] * p)
        pi = topkidx(-pw.dot(a), k1, inplace=True)
        ni = topkidx(nw.dot(a), k2, inplace=True)
    xpw = pw[pi]
    xnw = nw[ni]
    J = -xpw.dot(a).mean() + xnw.dot(a).mean()
    dW = -P[pi].T.dot(xp.tile(a, (k1, 1))) / k1 + N[ni].T.dot(xp.tile(a, (k2, 1))) / k2
    da = -xpw.mean(axis=0) + xnw.mean(axis=0)
    return J, dW, da


def ubise_full_ovo(P, N, a, W, p):
    J, dW, da = ubise(P, N, a, W, p)
    return J, dW, da, xp.zeros_like(da), xp.zeros_like(da), xp.zeros_like(da)


def ubise_full_ovr(P, Pn, SP, SPn, N, Nn, SN, SNn, a, c, e, g, W, p):
    J1, dW1, da = ubise(P, Pn, a, W, p)
    J2, dW2, dc = ubise(SP, SPn, c, W, p)
    J3, dW3, de = ubise(N, Nn, e, W, p)
    J4, dW4, dg = ubise(SN, SNn, g, W, p)
    return J1 + J2 + J3 + J4, dW1 + dW2 + dW3 + dW4, da, dc, de, dg


def proj_spectral(W, threshold):
    xp = get_array_module(W)
    u, s, vt = xp.linalg.svd(W)
    s[s > threshold] = threshold
    s[s < 0] = 0
    return xp.dot(u, xp.dot(xp.diag(s), vt))


def proj_l2(x, threshold=1):
    xp = get_array_module(x)
    return x / max(xp.linalg.norm(x) / threshold, 1)


def inspect_matrix(X):
    u, s, vt = xp.linalg.svd(X)
    print('mean singular value: {0:.4f}'.format(float(s.mean())))
    print('top 6 singular values: {0}'.format(str(s[:6])))


@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=UndefinedMetricWarning)
def main(args):
    print(str(args))

    # load source and target embeddings
    if args.pickle:
        with open(args.source_embedding, 'rb') as fin:
            src_wv = pickle.load(fin)
        with open(args.target_embedding, 'rb') as fin:
            trg_wv = pickle.load(fin)
    else:
        src_wv = WordVecs(args.source_embedding, emb_format=args.format).normalize(args.normalize)
        trg_wv = WordVecs(args.target_embedding, emb_format=args.format).normalize(args.normalize)

    # sentiment array
    src_pad_id = src_wv.add_word('<pad>', np.zeros(args.vector_dim, dtype=np.float32))
    src_ds = SentimentDataset(args.source_dataset).to_index(src_wv, binary=args.binary).pad(src_pad_id)
    trg_pad_id = trg_wv.add_word('<pad>', np.zeros(args.vector_dim, dtype=np.float32))
    trg_ds = SentimentDataset(args.target_dataset).to_index(trg_wv, binary=args.binary).pad(trg_pad_id)
    train_x, train_y, train_l = src_ds.train[0], src_ds.train[1], src_ds.train[2]
    dev_x = np.concatenate((trg_ds.train[0], trg_ds.dev[0]), axis=0)
    dev_y = np.concatenate((trg_ds.train[1], trg_ds.dev[1]), axis=0)
    dev_l = np.concatenate((trg_ds.train[2], trg_ds.dev[2]), axis=0)

    train_x = xp.array(train_x, dtype=xp.int32)
    train_y = xp.array(train_y, dtype=xp.int32)
    train_l = xp.array(train_l, dtype=xp.int32)
    dev_x = xp.array(dev_x, dtype=xp.int32)
    dev_y = xp.array(dev_y, dtype=xp.int32)
    dev_l = xp.array(dev_l, dtype=xp.int32)

    ys = np.concatenate((src_ds.train[1], trg_ds.train[1], trg_ds.dev[1]), axis=0)

    cv_split = np.zeros(train_x.shape[0] + dev_x.shape[0], dtype=np.int32)
    cv_split[:train_x.shape[0]] = -1
    cv_split = PredefinedSplit(cv_split)
    param_grid = {
        'C': [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000],
    }
    clf = GridSearchCV(svm.LinearSVC(), param_grid, scoring='f1_macro', n_jobs=cpu_count(), cv=cv_split)

    best_dev_f1 = 0
    with open(args.senti, 'rb') as fin:
        xsenti, ysenti = pickle.load(fin)

    xsenti = xp.array(xsenti, dtype=xp.float32)
    ysenti = xp.array(ysenti, dtype=xp.int32)
    P, SP = xsenti[ysenti == 0], xsenti[ysenti == 1]
    N, SN = xsenti[ysenti == 2], xsenti[ysenti == 3]
    PP = xp.concatenate((P, SP), axis=0)
    NN = xp.concatenate((N, SN), axis=0)
    Pn = xp.concatenate((SP, N, SN), axis=0)
    SPn = xp.concatenate((P, N, SN), axis=0)
    Nn = xp.concatenate((P, SP, SN), axis=0)
    SNn = xp.concatenate((P, SP, N), axis=0)

    if args.normalize_senti:
        length_normalize(xsenti, inplace=True)

    # prepare dictionaries
    gold_dict = xp.array(BilingualDict(args.gold_dictionary).get_indexed_dictionary(src_wv, trg_wv), dtype=xp.int32)
    init_dict = get_unsupervised_init_dict(src_wv.embedding, trg_wv.embedding, args.vocab_cutoff, args.csls, args.normalize, args.direction)
    init_dict = xp.array(init_dict)
    curr_dict = init_dict
    print('gold dict shape' + str(gold_dict.shape))

    # initialize hyper parameters
    keep_prob = args.dropout_init
    threshold = min(args.threshold, args.threshold_init)
    lr = args.learning_rate

    src_val_ind = np.union1d(asnumpy(gold_dict[:, 0]), train_x)
    # construct BDI object
    bdi_obj = BDI(src_wv.embedding, trg_wv.embedding, batch_size=args.batch_size, cutoff_size=args.vocab_cutoff, cutoff_type='both',
                  direction=args.direction, csls=args.csls, batch_size_val=args.val_batch_size, scorer=args.scorer,
                  src_val_ind=src_val_ind, trg_val_ind=gold_dict[:, 1])

    # print alignment error
    if not args.no_proj_error:
        proj_error = xp.sum((bdi_obj.src_proj_emb[gold_dict[:, 0]] - bdi_obj.trg_proj_emb[gold_dict[:, 1]])**2)
        print('proj error: %.4f' % proj_error)

    # initialize W_src and W_trg
    if args.load is not None:
        dic = load_model(args.load)
        W_src = xp.array(dic['W_source'], dtype=xp.float32)
        W_trg = xp.array(dic['W_target'], dtype=xp.float32)
    else:
        W_src = xp.identity(args.vector_dim, dtype=xp.float32)

        W_trg = xp.identity(args.vector_dim, dtype=xp.float32)
        lr = args.learning_rate

    bdi_obj.project(W_src, 'forward', unit_norm=args.normalize_projection)
    bdi_obj.project(W_trg, 'backward', unit_norm=args.normalize_projection, full_trg=True)

    # initialize model parameters
    a = xp.random.randn(args.vector_dim).astype(xp.float32)
    c = xp.random.randn(args.vector_dim).astype(xp.float32)
    e = xp.random.randn(args.vector_dim).astype(xp.float32)
    g = xp.random.randn(args.vector_dim).astype(xp.float32)
    a /= xp.linalg.norm(a)
    c /= xp.linalg.norm(c)
    e /= xp.linalg.norm(e)
    g /= xp.linalg.norm(g)

    # print alignment error
    if not args.no_proj_error:
        proj_error = xp.sum((bdi_obj.src_proj_emb[gold_dict[:, 0]] - bdi_obj.trg_proj_emb[gold_dict[:, 1]])**2)
        print('proj error: %.4f' % proj_error)

    # self learning
    try:
        for epoch in range(args.epochs):
            print()
            print('running epoch %d...' % epoch)
            print('threshold: %.4f' % threshold)

            # update W_src
            if epoch % 2 == 0:
                if args.model == '0':
                    continue

                # update a, c, e
                if args.model == 'ovo':
                    J, dW, da, dc, de, dg = ubise_full_ovo(PP, NN, a, W_src, args.p)
                elif args.model == 'ovr':
                    J, dW, da, dc, de, dg = ubise_full_ovr(P, Pn, SP, SPn, N, Nn, SN, SNn, a, c, e, g, W_src, args.p)
                print('\rJ: {0:.10f}'.format(float(J)), end='')
                lr = args.learning_rate
                cnt = 0
                while lr > 1e-4:
                    Jold, Wold, aold, cold, eold, gold = J, W_src.copy(), a.copy(), c.copy(), e.copy(), g.copy()
                    dWold, daold, dcold, deold, dgold = dW.copy(), da.copy(), dc.copy(), de.copy(), dg.copy()
                    a, c, e, g = proj_l2(a - lr * da), proj_l2(c - lr * dc), proj_l2(e - lr * de), proj_l2(g - lr * g)
                    if args.model == 'ovo':
                        J, dW, da, dc, de, dg = ubise_full_ovo(PP, NN, a, W_src, args.p)
                    elif args.model == 'ovr':
                        J, dW, da, dc, de, dg = ubise_full_ovr(P, Pn, SP, SPn, N, Nn, SN, SNn, a, c, e, g, W_src, args.p)
                    print('\rJ: {0:.10f}'.format(float(J)), end='')
                    if J > Jold:
                        lr /= 2
                        J, W_src, a, c, e, g = Jold, Wold, aold, cold, eold, gold
                        dW, da, dc, de, dg = dWold, daold, dcold, deold, dgold
                    elif Jold - J < 0.0000001:
                        break
                print()

                # update W_src
                if args.model == 'ovo':
                    J, dW, da, dc, de, dg = ubise_full_ovo(PP, NN, a, W_src, args.p)
                elif args.model == 'ovr':
                    J, dW, da, dc, de, dg = ubise_full_ovr(P, Pn, SP, SPn, N, Nn, SN, SNn, a, c, e, g, W_src, args.p)
                print('\rJ: {0:.10f}'.format(float(J)), end='')
                lr = args.learning_rate
                cnt = 0
                while lr > 1e-10:
                    Jold, Wold, aold, cold, eold, gold = J, W_src.copy(), a.copy(), c.copy(), e.copy(), g.copy()
                    dWold, daold, dcold, deold, dgold = dW.copy(), da.copy(), dc.copy(), de.copy(), dgold.copy()
                    W_src = proj_spectral(W_src - lr * dW, threshold=threshold)
                    if args.model == 'ovo':
                        J, dW, da, dc, de, dg = ubise_full_ovo(PP, NN, a, W_src, args.p)
                    elif args.model == 'ovr':
                        J, dW, da, dc, de, dg = ubise_full_ovr(P, Pn, SP, SPn, N, Nn, SN, SNn, a, c, e, g, W_src, args.p)
                    print('\rJ: {0:.10f}'.format(float(J)), end='')
                    if J > Jold:
                        lr /= 2
                        J, W_src, a, c, e, g = Jold, Wold, aold, cold, eold, gold
                        dW, da, dc, de, dg = dWold, daold, dcold, deold, dgold
                    elif Jold - J < 0.0000001:
                        break
                print()
                inspect_matrix(W_src)
                bdi_obj.project(W_src, 'forward', unit_norm=args.normalize_projection)

            # update W_trg
            elif epoch % 2 == 1:
                if args.target_loss == 'procruste':
                    # procruste
                    lr = args.learning_rate
                    X_src = bdi_obj.src_proj_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_emb[curr_dict[:, 1]]

                    W_trg = xp.linalg.pinv(X_trg).dot(X_src)  # procruste initialization
                    W_trg = proj_spectral(W_trg, threshold=1)

                    loss = xp.linalg.norm(X_trg.dot(W_trg) - X_src)**2
                    while lr > 0.000000005:
                        prev_W = W_trg.copy()
                        prev_loss = loss
                        grad = 2 * (X_trg.T.dot(X_trg).dot(W_trg) - X_trg.T.dot(X_src))
                        W_trg -= lr * grad
                        W_trg = proj_spectral(W_trg, threshold=threshold)
                        loss = xp.linalg.norm(X_trg.dot(W_trg) - X_src)**2
                        if loss > prev_loss:
                            lr /= 2
                            W_trg = prev_W
                            loss = prev_loss
                        elif prev_loss - loss < 1e-2:
                            break
                        print('\rloss: {0:.4f}'.format(float(loss)), end='')
                    print()

                if args.target_loss == 'orthogonal':
                    # procruste
                    lr = args.learning_rate
                    X_src = bdi_obj.src_proj_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_emb[curr_dict[:, 1]]

                    u, s, vt = xp.linalg.svd(X_trg.T.dot(X_src))  # procruste initialization
                    W_trg = u.dot(vt)

                    loss = -(X_trg.dot(W_trg) * X_src).sum()
                    while lr > 0.000000005:
                        prev_W = W_trg.copy()
                        prev_loss = loss
                        grad = 2 * X_trg.T.dot(X_src)
                        W_trg -= lr * grad
                        W_trg = proj_spectral(W_trg, threshold=threshold)
                        loss = -(X_trg.dot(W_trg) * X_src).sum()
                        if loss > prev_loss:
                            lr /= 2
                            W_trg = prev_W
                            loss = prev_loss
                        elif prev_loss - loss < 1e-2:
                            break
                        print('\rloss: {0:.4f}'.format(float(loss)), end='')
                    print()

                elif args.target_loss == 'whitten':
                    X_src = bdi_obj.src_proj_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_emb[curr_dict[:, 1]]
                    u, s, vt = xp.linalg.svd(X_src)
                    W_src_1 = vt.T.dot(xp.diag(1 / s)).dot(vt)
                    X_src_1 = X_src.dot(W_src_1)

                    u, s, vt = xp.linalg.svd(X_trg)
                    W_trg_1 = vt.T.dot(xp.diag(1 / s)).dot(vt)
                    X_trg_1 = X_trg.dot(W_trg_1)

                    u, s, vt = xp.linalg.svd(X_trg_1.T.dot(X_src_1))
                    W_trg_2 = u.dot(vt)

                    W_trg = W_trg_1.dot(W_trg_2).dot(xp.linalg.pinv(W_src_1))
                    W_trg = proj_spectral(W_trg, threshold=threshold)

                inspect_matrix(W_trg)
                bdi_obj.project(W_trg, 'backward', unit_norm=args.normalize_projection, full_trg=True)

            if not args.no_proj_error:
                proj_error = xp.sum((bdi_obj.src_proj_emb[gold_dict[:, 0]] - bdi_obj.trg_proj_emb[gold_dict[:, 1]])**2)
                print('proj error: %.4f' % proj_error)

            # update current dictionary
            if epoch % 2 == 1 and threshold < args.threshold_valid:
                curr_dict = bdi_obj.get_bilingual_dict_with_cutoff(keep_prob=keep_prob)

            # update keep_prob
            keep_prob = min(1., keep_prob + args.dropout_step)

            # update threshold
            if epoch % 2 == 1:
                threshold = min(args.threshold_step * 2 + threshold, args.threshold)

                if args.dump and threshold % 0.5 < args.threshold_step * 2:
                    export_path = args.save_path.split('-')
                    for i in range(len(export_path)):
                        if export_path[i][0] == 't':
                            export_path[i] = 't{:.1f}'.format(threshold)
                    export_path = '-'.join(export_path)
                    model = 'ubise' if args.normalize_projection else args.model
                    save_model(asnumpy(W_src), asnumpy(W_trg), args.source_lang,
                               args.target_lang, model, export_path,
                               alpha=args.alpha, dropout_init=args.dropout_init,
                               a=asnumpy(a), c=asnumpy(c), e=asnumpy(e), g=asnumpy(g))

            # valiadation
            if not args.no_valiadation and (epoch + 1) % args.valiadation_step == 0 or epoch == (args.epochs - 1):
                bdi_obj.project(W_trg, 'backward', unit_norm=args.normalize_projection, full_trg=True)
                val_trg_ind = bdi_obj.get_target_indices(gold_dict[:, 0])
                accuracy = xp.mean((val_trg_ind == gold_dict[:, 1]).astype(xp.int32))
                print('epoch: %d   accuracy: %.4f   dict_size: %d' % (epoch, accuracy, curr_dict.shape[0]))
    finally:
        # save W_src and W_trg
        if args.spectral:
            W_src = proj_spectral(W_src, threshold=args.threshold)
            W_trg = proj_spectral(W_trg, threshold=args.threshold)
        model = 'ubise' if args.normalize_projection else args.model
        save_model(asnumpy(W_src), asnumpy(W_trg), args.source_lang,
                   args.target_lang, model, args.save_path,
                   alpha=args.alpha, dropout_init=args.dropout_init,
                   a=asnumpy(a), c=asnumpy(c), e=asnumpy(e), g=asnumpy(g))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_loss', choices=['procruste', 'whitten', 'orthogonal'], default='procruste', help='target loss function')
    parser.add_argument('--sample', choices=['uniform', 'smooth'], default='uniform', help='sampling method')
    parser.add_argument('--smooth', type=int, default=0.5, help='smoothing power')
    parser.add_argument('--normalize_senti', action='store_true', help='l2-normalize sentiment vectors')
    parser.add_argument('-p', '--p', type=float, default=0.7, help='parameter p')
    parser.add_argument('-k', '--k', type=int, default=10, help='parameter k')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--model', choices=['ovo', 'ovr', '0'], default='ovr', help='source objective function')
    parser.add_argument('--scorer', choices=['dot', 'euclidean'], default='dot', help='retrieval method')
    parser.add_argument('--senti', default='pickle/senti_opener.bin', help='sentiment vectors')
    parser.add_argument('--dump', action='store_true')
    parser.add_argument('-bi', '--binary', action='store_true')

    training_group = parser.add_argument_group()
    training_group.add_argument('--source_lang', default='en', help='source language')
    training_group.add_argument('--target_lang', default='es', help='target language')
    training_group.add_argument('-e', '--epochs', default=500, type=int, help='training epochs (default: 500)')
    training_group.add_argument('-bs', '--batch_size', default=3000, type=int, help='training batch size (default: 3000)')
    training_group.add_argument('-vbs', '--val_batch_size', default=300, type=int, help='training batch size (default: 300)')
    training_group.add_argument('--no_valiadation', action='store_true', help='disable valiadation at each iteration')
    training_group.add_argument('--no_proj_error', action='store_true', help='disable proj error monitoring')
    training_group.add_argument('--valiadation_step', type=int, default=50, help='valiadation frequency')
    training_group.add_argument('--debug', action='store_const', dest='loglevel', default=logging.INFO, const=logging.DEBUG, help='print debug info')
    training_group.add_argument('--cuda', action='store_true', help='use cuda to accelerate')

    dataset_group = parser.add_argument_group()
    dataset_group.add_argument('-se', '--source_embedding', default='./emb/en.bin', help='monolingual word embedding of the source language (default: ./emb/en.bin)')
    dataset_group.add_argument('-te', '--target_embedding', default='./emb/es.bin', help='monolingual word embedding of the target language (default: ./emb/es.bin)')
    dataset_group.add_argument('-vd', '--vector_dim', default=300, type=int, help='dimension of each word vector (default: 300)')
    dataset_group.add_argument('--format', choices=['word2vec_bin', 'fasttext_text'], default='word2vec_bin', help='word embedding format')
    dataset_group.add_argument('-sd', '--source_dataset', default='./datasets/en/opener_sents/', help='source sentiment dataset')
    dataset_group.add_argument('-gd', '--gold_dictionary', default='./lexicons/apertium/en-es.txt', help='gold bilingual dictionary for evaluation(default: ./lexicons/apertium/en-es.txt)')

    io_group = parser.add_argument_group()
    io_group.add_argument('--load', help='restore W_src and W_trg from a file')
    io_group.add_argument('--pickle', action='store_true', help='load from pickled objects')
    io_group.add_argument('--save_path', default='./checkpoints/senti.bin', help='file to save W_src and W_trg')

    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument('--init_unsupervised', action='store_true', help='use unsupervised init')

    mapping_group = parser.add_argument_group()
    mapping_group.add_argument('--normalize', choices=['unit', 'center', 'unitdim', 'centeremb', 'none'], nargs='*', default=['center', 'unit'], help='normalization actions')
    mapping_group.add_argument('--spectral', action='store_true', help='restrict projection matrix to spectral domain')
    mapping_group.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='use gradient descent to solve W')
    mapping_group.add_argument('--normalize_projection', action='store_true', help='normalize after projection')

    threshold_group = parser.add_argument_group()
    threshold_group.add_argument('--threshold', type=float, default=1., help='spectral norm constraint')
    threshold_group.add_argument('--threshold_init', type=float, default=1., help='spectral norm constraint')
    threshold_group.add_argument('--threshold_step', type=float, default=0.05, help='spectral norm constraint')
    threshold_group.add_argument('--threshold_valid', type=float, default=float('inf'), help='valid threshold for updating bilingual dictionary')

    induction_group = parser.add_argument_group()
    induction_group.add_argument('-vc', '--vocab_cutoff', default=10000, type=int, help='restrict the vocabulary to k most frequent words')
    induction_group.add_argument('--csls', type=int, default=10, help='number of csls neighbours')
    induction_group.add_argument('--dropout_init', type=float, default=0.1, help='initial keep prob of the dropout machanism')
    induction_group.add_argument('--dropout_step', type=float, default=0.1, help='increase keep_prob by a small step')
    induction_group.add_argument('--direction', choices=['forward', 'backward', 'union'], default='union', help='direction of dictionary induction')

    recommend_group = parser.add_mutually_exclusive_group()
    recommend_group.add_argument('-u', '--unsupervised', action='store_true', help='use recommended settings')

    lang_group = parser.add_mutually_exclusive_group()
    lang_group.add_argument('--en_es', action='store_true', help='train english-spanish embedding')
    lang_group.add_argument('--en_ca', action='store_true', help='train english-catalan embedding')
    lang_group.add_argument('--en_eu', action='store_true', help='train english-basque embedding')
    lang_group.add_argument('--en_fr', action='store_true', help='train english-french embedding')
    lang_group.add_argument('--en_de', action='store_true', help='train english-german embedding')
    lang_group.add_argument('--en_ja', action='store_true', help='train english-japanese embedding')

    args = parser.parse_args()
    parser.set_defaults(init_unsupervised=True, csls=10, direction='union', cuda=False, normalize=['center', 'unit'],
                        vocab_cutoff=10000, alpha=1., spectral=True,
                        learning_rate=10000, save_path='checkpoints/ubise.bin',
                        dropout_init=0.1, dropout_step=0.002, epochs=500, normalize_projection=False,
                        threshold=1.0, batch_size=5000, val_batch_size=300)

    if args.en_es:
        src_emb_file = 'pickle/en.bin' if args.pickle else 'emb/wiki.en.vec'
        trg_emb_file = 'pickle/es.bin' if args.pickle else 'emb/wiki.es.vec'
        parser.set_defaults(source_lang='en', target_lang='es',
                            source_embedding=src_emb_file, target_embedding=trg_emb_file, format='fasttext_text',
                            source_dataset='datasets/en/opener_sents/', target_dataset='datasets/es/opener_sents/',
                            gold_dictionary='lexicons/muse/en-es.0-5000.txt')
    elif args.en_ca:
        src_emb_file = 'pickle/en.bin' if args.pickle else 'emb/wiki.en.vec'
        trg_emb_file = 'pickle/ca.bin' if args.pickle else 'emb/wiki.ca.vec'
        parser.set_defaults(source_lang='en', target_lang='ca',
                            source_embedding=src_emb_file, target_embedding=trg_emb_file, format='fasttext_text',
                            source_dataset='datasets/en/opener_sents/', target_dataset='datasets/ca/opener_sents/',
                            gold_dictionary='lexicons/muse/en-ca.0-5000.txt')
    elif args.en_eu:
        src_emb_file = 'pickle/en.bin' if args.pickle else 'emb/wiki.en.vec'
        trg_emb_file = 'pickle/eu.bin' if args.pickle else 'emb/wiki.eu.vec'
        parser.set_defaults(source_lang='en', target_lang='eu',
                            source_embedding=src_emb_file, target_embedding=trg_emb_file, format='fasttext_text',
                            source_dataset='datasets/en/opener_sents/', target_dataset='datasets/eu/opener_sents/',
                            gold_dictionary='lexicons/muse/en-eu.0-5000.txt')

    elif args.en_fr:
        src_emb_file = 'pickle/en.bin' if args.pickle else 'emb/wiki.en.vec'
        trg_emb_file = 'pickle/fr.bin' if args.pickle else 'emb/wiki.fr.vec'
        parser.set_defaults(source_lang='en', target_lang='fr',
                            source_embedding=src_emb_file, target_embedding=trg_emb_file, format='fasttext_text',
                            source_dataset='datasets/cls10/en/books/',
                            gold_dictionary='lexicons/muse/en-fr.0-5000.txt')

    elif args.en_de:
        src_emb_file = 'pickle/en.bin' if args.pickle else 'emb/wiki.en.vec'
        trg_emb_file = 'pickle/de.bin' if args.pickle else 'emb/wiki.de.vec'
        parser.set_defaults(source_lang='en', target_lang='de',
                            source_embedding=src_emb_file, target_embedding=trg_emb_file, format='fasttext_text',
                            source_dataset='datasets/cls10/en/books/',
                            gold_dictionary='lexicons/muse/en-de.0-5000.txt')

    elif args.en_ja:
        src_emb_file = 'pickle/en.bin' if args.pickle else 'emb/wiki.en.vec'
        trg_emb_file = 'pickle/ja.bin' if args.pickle else 'emb/wiki.ja.vec'
        parser.set_defaults(source_lang='en', target_lang='ja',
                            source_embedding=src_emb_file, target_embedding=trg_emb_file, format='fasttext_text',
                            source_dataset='datasets/cls10/en/books/',
                            gold_dictionary='lexicons/muse/en-ja.0-5000.txt')

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel, format='%(asctime)s: %(message)s')

    if args.cuda:
        xp = get_cupy()
        if xp is None:
            print('Install cupy for cuda support')
            sys.exit(-1)
    else:
        xp = np

    xp.random.seed(args.seed)

    main(args)
