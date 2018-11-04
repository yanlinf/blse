import argparse
import pickle
import logging
import sys
import os
import re
import numpy as np
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
    k1 = int(P.shape[0] * p)
    k2 = int(N.shape[0] * p)
    pi = topkidx(-pw.dot(a), k1, inplace=True)
    ni = topkidx(nw.dot(a), k2, inplace=True)
    xpw = pw[pi]
    xnw = nw[ni]
    J = -xpw.dot(a).mean() + xnw.dot(a).mean()
    dW = -xpw.T.dot(xp.tile(a, (k1, 1))) / k1 + xnw.T.dot(xp.tile(a, (k2, 1))) / k2
    da = W.T.dot(-xpw.mean(axis=0) + xnw.mean(axis=0))
    return J, dW, da


def proj_spectral(W, threshold):
    xp = get_array_module(W)
    u, s, vt = xp.linalg.svd(W)
    s[s > threshold] = threshold
    s[s < 0] = 0
    return xp.dot(u, xp.dot(xp.diag(s), vt))


def inspect_matrix(X):
    u, s, vt = xp.linalg.svd(X)
    logging.debug('Squared F-norm: {0:.4f}'.format(float((X**2).sum())))
    logging.debug('Spectral norm: {0:.4f}'.format(float(s[0])))
    logging.debug('10th maximum singular value: {0:.4f}'.format(float(s[10])))
    logging.debug('mean singular value: {0:.4f}'.format(float(s.mean())))
    logging.debug('top 6 singular values: {0}'.format(str(s[:6])))


def main(args):
    logging.info(str(args))

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
    pad_id = src_wv.add_word('<pad>', np.zeros(args.vector_dim, dtype=np.float32))
    src_ds = SentimentDataset(args.source_dataset).to_index(src_wv, binary=False).pad(pad_id)
    xsenti = xp.array(src_wv.embedding[src_ds.train[0]].sum(axis=1) / src_ds.train[2][:, np.newaxis], dtype=xp.float32)
    ysenti = xp.array(src_ds.train[1], dtype=xp.int32)

    if args.fine_grained:
        n = ysenti.shape[0]
        n0 = (ysenti == 0).sum()
        n1 = (ysenti == 1).sum()
        n2 = (ysenti == 2).sum()
        n3 = (ysenti == 3).sum()
        m = args.senti_nsample
        ms = xp.array([n0, n1, n2, n3])
        if args.sample == 'smooth':
            ms = ms**args.smooth
        ms = ms * m / ms.sum()
        ms = ms.astype(xp.int32)
        m0, m1, m2, _ = ms
        m3 = m - m0 - m1 - m2
        logging.debug(str((n0, n1, n2, n3)))
        logging.debug(str((m0, m1, m2, m3)))
        sa = xp.arange(n, dtype=xp.int32)

    if args.normalize_senti:
        length_normalize(xsenti, inplace=True)

    # prepare dictionaries
    gold_dict = xp.array(BilingualDict(args.gold_dictionary).get_indexed_dictionary(src_wv, trg_wv), dtype=xp.int32)
    init_dict = get_unsupervised_init_dict(src_wv.embedding, trg_wv.embedding, args.vocab_cutoff, args.csls, args.normalize, args.direction)
    init_dict = xp.array(init_dict)
    logging.debug('gold dict shape' + str(gold_dict.shape))

    # initialize hyper parameters
    keep_prob = args.dropout_init
    alpha = min(args.alpha, args.alpha_init)
    threshold = min(args.threshold, args.threshold_init)

    # construct BDI object
    bdi_obj = BDI(src_wv.embedding, trg_wv.embedding, batch_size=args.batch_size, cutoff_size=args.vocab_cutoff, cutoff_type='both',
                  direction=args.direction, csls=args.csls, batch_size_val=args.val_batch_size, scorer='dot',
                  src_val_ind=gold_dict[:, 0], trg_val_ind=gold_dict[:, 1])

    # print alignment error
    if not args.no_proj_error:
        proj_error = xp.sum((bdi_obj.src_proj_emb[gold_dict[:, 0]] - bdi_obj.trg_proj_emb[gold_dict[:, 1]])**2)
        logging.info('proj error: %.4f' % proj_error)

    # initialize W_src and W_trg
    if args.load is not None:
        dic = load_model(args.load)
        W_src = xp.array(dic['W_source'], dtype=xp.float32)
        W_trg = xp.array(dic['W_target'], dtype=xp.float32)
    else:
        W_src = xp.identity(args.vector_dim, dtype=xp.float32)

        W_trg = xp.identity(args.vector_dim, dtype=xp.float32)
        lr = args.learning_rate
        X_src = bdi_obj.src_proj_emb[init_dict[:, 0]]
        X_trg = bdi_obj.trg_emb[init_dict[:, 1]]
        prev_loss, loss = float('inf'), float('inf')
        while lr > 0.00006:
            prev_W = W_trg.copy()
            prev_loss = loss
            grad = -2 * X_trg.T.dot(X_src)
            W_trg -= lr * grad
            W_trg = proj_spectral(W_trg, threshold=threshold)
            loss = -2 * (X_trg.dot(W_trg) * X_src).sum()
            if loss > prev_loss:
                lr /= 2
                W_trg = prev_W
                loss = prev_loss
            elif prev_loss - loss < 0.5:
                break
            u, s, vt = xp.linalg.svd(W_trg)
            W_trg = u.dot(vt)

    bdi_obj.project(W_src, 'forward', unit_norm=args.normalize_projection)
    bdi_obj.project(W_trg, 'backward', unit_norm=args.normalize_projection, full_trg=True)

    # initialize model parameters
    if args.fine_grained:
        a = xp.zeros(args.vector_dim, dtype=xp.float32)
    else:
        a = xp.random.randn(args.vector_dim).astype(xp.float32)
    c = xp.zeros(args.vector_dim, dtype=xp.float32)
    e = xp.zeros(args.vector_dim, dtype=xp.float32)
    b = xp.zeros((), dtype=xp.float32)
    d = xp.zeros((), dtype=xp.float32)
    f = xp.zeros((), dtype=xp.float32)

    # print alignment error
    if not args.no_proj_error:
        proj_error = xp.sum((bdi_obj.src_proj_emb[gold_dict[:, 0]] - bdi_obj.trg_proj_emb[gold_dict[:, 1]])**2)
        logging.info('proj error: %.4f' % proj_error)

    # self learning
    try:
        for epoch in range(args.epochs):
            logging.debug('running epoch %d...' % epoch)
            logging.debug('alpha: %.4f' % alpha)
            logging.debug('threshold: %.4f' % threshold)

            # update current dictionary
            if epoch % 2 == 0:
                curr_dict = bdi_obj.get_bilingual_dict_with_cutoff(keep_prob=keep_prob)

            # update W_src
            if epoch % 2 == 0:
                m = args.senti_nsample
                lr = args.learning_rate
                X_src = bdi_obj.src_emb[curr_dict[:, 0]]
                X_trg = bdi_obj.trg_proj_emb[curr_dict[:, 1]]

                if args.fine_grained:
                    if args.sample == 'uniform':
                        sind = xp.random.randint(0, xsenti.shape[0], m)
                    elif args.sample == 'smooth':
                        si0 = sa[ysenti == 0][xp.random.randint(0, n0, m0)]
                        si1 = sa[ysenti == 1][xp.random.randint(0, n1, m1)]
                        si2 = sa[ysenti == 2][xp.random.randint(0, n2, m2)]
                        si3 = sa[ysenti == 3][xp.random.randint(0, n3, m3)]
                        sind = xp.concatenate((si0, si1, si2, si3), axis=0)

                    Xs = xsenti[sind]
                    ys = (ysenti[sind] <= 1).astype(xp.float32) * 2 - 1  # 1 = positive, -1 = negative
                    ts = xp.ones(m, dtype=xp.float32)  # weights

                    spi = sa[ysenti == 1][xp.random.randint(0, n1, m // 2)]
                    spin = sa[ysenti != 1][xp.random.randint(0, n0 + n2 + n3, (m + 1) // 2)]  # m // 2 + (m + 1) // 2 == m
                    sni = sa[ysenti == 3][xp.random.randint(0, n3, m // 2)]
                    snin = sa[ysenti != 3][xp.random.randint(0, n0 + n1 + n2, (m + 1) // 2)]

                    pind = xp.concatenate((spi, spin), axis=0)
                    nind = xp.concatenate((sni, snin), axis=0)
                    Xp = xsenti[pind]
                    Xn = xsenti[nind]
                    yp = (ysenti[pind] == 1).astype(xp.float32) * 2 - 1
                    yn = (ysenti[nind] == 3).astype(xp.float32) * 2 - 1

                    loss = (1000 / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(a)) + b) * ys).dot(ts).sum() +\
                        (alpha / m) * (xp.maximum(0, 1 - (Xp.dot(W_src.dot(c)) + d) * yp).sum() +
                                       xp.maximum(0, 1 - (Xn.dot(W_src.dot(e)) + f) * yn).sum())
                    logging.debug('loss: {0:.4f}'.format(float(loss)))
                    cnt = 0
                    while lr > 0.0000000000000005:
                        prev_W = W_src.copy()
                        prev_a = a.copy()
                        prev_b = b.copy()
                        prev_c = c.copy()
                        prev_d = d.copy()
                        prev_e = e.copy()
                        prev_f = f.copy()
                        prev_loss = loss

                        masks = ((Xs.dot(W_src.dot(a)) + b) * ys < 1).astype(xp.float32)
                        maskp = ((Xp.dot(W_src.dot(c)) + d) * yp < 1).astype(xp.float32)
                        maskn = ((Xn.dot(W_src.dot(e)) + f) * yn < 1).astype(xp.float32)

                        Zs = Xs * (-ys * masks)[:, xp.newaxis]
                        Zp = Xp * (-yp * maskp)[:, xp.newaxis]
                        Zn = Xn * (-yn * maskn)[:, xp.newaxis]

                        dW = (1000 / m) * Zs.T.dot(xp.tile(a, (m, 1))) +\
                            (alpha / m) * (Zp.T.dot(xp.tile(c, (m, 1))) +
                                           Zn.T.dot(xp.tile(e, (m, 1))))
                        da = (1000 / m) * W_src.T.dot(Zs.sum(axis=0))
                        db = (1000 / m) * (-ys * masks).sum()
                        dc = (alpha / m) * W_src.T.dot(Zp.sum(axis=0))
                        dd = (alpha / m) * (-yp * maskp).sum()
                        de = (alpha / m) * W_src.T.dot(Zn.sum(axis=0))
                        df = (alpha / m) * (-yn * maskn).sum()

                        W_src = proj_spectral(W_src - lr * dW, threshold=threshold).astype(xp.float32)
                        a -= lr * da
                        b -= lr * db
                        c -= lr * dc
                        d -= lr * dd
                        e -= lr * de
                        f -= lr * df

                        loss = (1000 / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(a)) + b) * ys).dot(ts).sum() +\
                            (alpha / m) * (xp.maximum(0, 1 - (Xp.dot(W_src.dot(c)) + d) * yp).sum() +
                                           xp.maximum(0, 1 - (Xn.dot(W_src.dot(e)) + f) * yn).sum())
                        logging.debug('loss: {0:.4f}'.format(float(loss)))
                        if loss >= prev_loss:
                            lr /= 2
                            W_src, a, b, c, d, e, f = prev_W, prev_a, prev_b, prev_c, prev_d, prev_e, prev_f
                            loss = prev_loss
                        else:
                            cnt += 1
                            if cnt == 10:
                                break
                    logging.debug('activated number %d' % int(masks.sum() + maskp.sum() + maskn.sum()))

                else:
                    P = xsenti[ysenti == 0]
                    N = xsenti[ysenti == 2]
                    loss, dW, da = ubise(P, N, a, W_src, args.p)
                    logging.debug('loss: {0:.10f}'.format(float(loss)))
                    cnt = 0
                    while lr > 1e-30:
                        prev_loss = loss
                        prev_dW = dW.copy()
                        prev_da = da.copy()
                        prev_W = W_src.copy()
                        prev_a = a.copy()
                        W_src = proj_spectral(W_src - lr * dW, threshold=threshold)
                        a -= lr * da
                        loss, dW, da = ubise(P, N, a, W_src, args.p)
                        logging.debug('loss: {0:.10f}'.format(float(loss)))
                        if loss >= prev_loss:
                            lr /= 2
                            W_src, a = prev_W, prev_a,
                            dW, da = prev_dW, prev_da
                            loss = prev_loss
                        else:
                            cnt += 1
                            if cnt == 10:
                                break

                inspect_matrix(W_src)
                bdi_obj.project(W_src, 'forward', unit_norm=args.normalize_projection)

            # update W_trg
            elif epoch % 2 == 1:
                if args.target_loss == 'procruste':
                    # procruste
                    lr = args.learning_rate
                    X_src = bdi_obj.src_proj_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_emb[curr_dict[:, 1]]
                    prev_loss, loss = float('inf'), float('inf')
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
                        elif prev_loss - loss < 1:
                            break
                        logging.debug('loss: {0:.4f}'.format(float(loss)))

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
                bdi_obj.project(W_trg, 'backward', unit_norm=args.normalize_projection)

            if not args.no_proj_error:
                proj_error = xp.sum((bdi_obj.src_proj_emb[gold_dict[:, 0]] - bdi_obj.trg_proj_emb[gold_dict[:, 1]])**2)
                logging.info('proj error: %.4f' % proj_error)

            # update keep_prob
            keep_prob = min(1., keep_prob + args.dropout_step)

            # update alpha
            alpha = min(args.alpha_step + alpha, args.alpha)

            # update threshold
            threshold = min(args.threshold_step + threshold, args.threshold)

            # valiadation
            if not args.no_valiadation and (epoch + 1) % args.valiadation_step == 0 or epoch == (args.epochs - 1):
                bdi_obj.project(W_trg, 'backward', unit_norm=True, full_trg=True)
                val_trg_ind = bdi_obj.get_target_indices(gold_dict[:, 0])
                accuracy = xp.mean((val_trg_ind == gold_dict[:, 1]).astype(xp.int32))
                logging.info('epoch: %d   accuracy: %.4f   dict_size: %d' % (epoch, accuracy, curr_dict.shape[0]))
    finally:
        # save W_trg
        if args.spectral:
            W_src = proj_spectral(W_src, threshold=args.threshold)
            W_trg = proj_spectral(W_trg, threshold=args.threshold)
        save_model(asnumpy(W_src), asnumpy(W_trg), args.source_lang,
                   args.target_lang, args.model, args.save_path,
                   alpha=args.alpha, alpha_init=args.alpha_init,
                   dropout_init=args.dropout_init,
                   a=asnumpy(a), b=asnumpy(b),
                   c=asnumpy(c), d=asnumpy(d),
                   e=asnumpy(e), f=asnumpy(f))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_loss', choices=['procruste', 'whitten'], default='procruste', help='target loss function')
    parser.add_argument('--sample', choices=['uniform', 'smooth'], default='uniform', help='sampling method')
    parser.add_argument('--smooth', type=int, default=0.5, help='smoothing power')
    parser.add_argument('--fine_grained', action='store_true', help='add fine grained loss term')
    parser.add_argument('--normalize_senti', action='store_true', help='l2-normalize sentiment vectors')
    parser.add_argument('-p', '--p', type=float, help='parameter p')

    training_group = parser.add_argument_group()
    training_group.add_argument('--source_lang', default='en', help='source language')
    training_group.add_argument('--target_lang', default='es', help='target language')
    training_group.add_argument('--model', choices=['ubi', 'ubise'], help='model type')
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

    senti_sample_group = parser.add_argument_group()
    senti_sample_group.add_argument('-n', '--senti_nsample', type=int, default=200, help='sentiment examples')
    senti_sample_group.add_argument('--sample_type', choices=['full', 'same', 'pos-neg'], default='full', help='positive examples')

    alpha_group = parser.add_argument_group()
    alpha_group.add_argument('-a', '--alpha', type=float, default=5, help='trade-off between sentiment and alignment')
    alpha_group.add_argument('--alpha_init', type=float, default=0., help='initial value of alpha')
    alpha_group.add_argument('--alpha_step', type=float, default=0.02, help='multiply alpha by a factor each epoch')

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

    args = parser.parse_args()
    parser.set_defaults(init_unsupervised=True, csls=10, direction='union', cuda=False, normalize=['center', 'unit'],
                        vocab_cutoff=10000, alpha=5000, senti_nsample=50, spectral=True,
                        learning_rate=0.01, alpha_init=5000, alpha_step=0.01, alpha_inc=True,
                        no_proj_error=False, save_path='checkpoints/cvxse.bin',
                        dropout_init=0.1, dropout_interval=1, dropout_step=0.002, epochs=1000,
                        no_target_senti=True, model='ubise', normalize_projection=False,
                        threshold=1.0,
                        batch_size=5000, val_batch_size=300)

    if args.en_es:
        src_emb_file = 'pickle/en.bin' if args.pickle else 'emb/wiki.en.vec'
        trg_emb_file = 'pickle/es.bin' if args.pickle else 'emb/wiki.es.vec'
        parser.set_defaults(source_lang='en', target_lang='es',
                            source_embedding=src_emb_file, target_embedding=trg_emb_file, format='fasttext_text',
                            source_dataset='datasets/en/opener_sents/', target_dataset='datasets/es/opener_sents/',
                            gold_dictionary='lexicons/apertium/en-es.txt')
    elif args.en_ca:
        src_emb_file = 'pickle/en.bin' if args.pickle else 'emb/wiki.en.vec'
        trg_emb_file = 'pickle/ca.bin' if args.pickle else 'emb/wiki.ca.vec'
        parser.set_defaults(source_lang='en', target_lang='ca',
                            source_embedding=src_emb_file, target_embedding=trg_emb_file, format='fasttext_text',
                            source_dataset='datasets/en/opener_sents/', target_dataset='datasets/ca/opener_sents/',
                            gold_dictionary='lexicons/apertium/en-ca.txt')
    elif args.en_eu:
        src_emb_file = 'pickle/en.bin' if args.pickle else 'emb/wiki.en.vec'
        trg_emb_file = 'pickle/eu.bin' if args.pickle else 'emb/wiki.eu.vec'
        parser.set_defaults(source_lang='en', target_lang='eu',
                            source_embedding=src_emb_file, target_embedding=trg_emb_file, format='fasttext_text',
                            source_dataset='datasets/en/opener_sents/', target_dataset='datasets/eu/opener_sents/',
                            gold_dictionary='lexicons/apertium/en-eu.txt')

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel, format='%(asctime)s: %(message)s')

    if args.cuda:
        xp = get_cupy()
        if xp is None:
            print('Install cupy for cuda support')
            sys.exit(-1)
    else:
        xp = np

    main(args)
