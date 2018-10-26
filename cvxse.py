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


def proj_spectral(W, tanh=False, threshold=1.):
    xp = get_array_module(W)
    u, s, vt = xp.linalg.svd(W)
    if tanh:
        s = xp.tanh(s)
    else:
        s[s > threshold] = threshold
        s[s < 0] = 0
    # DEBUG(s)
    return xp.dot(u, xp.dot(xp.diag(s), vt))


def getknn(sc, x, y, k=10):
    xp = get_array_module(sc, x, y)
    sidx = xp.empty((sc.shape[0], k), dtype=xp.int32)
    for i in range(0, sc.shape[0], 1000):
        j = min(i + 1000, sc.shape[0])
        sidx[i:j] = xp.argpartition(sc[i:j], -k, axis=1)[:, -k:]
    # sidx = xp.argpartition(sc, -k, axis=1)[:, -k:]
    DEBUG(3)
    ytopk = y[sidx.flatten(), :]
    DEBUG(4)
    ytopk = ytopk.reshape(sidx.shape[0], sidx.shape[1], y.shape[1])
    DEBUG(5)
    f = xp.sum(sc[xp.arange(sc.shape[0])[:, None], sidx])
    DEBUG(6)
    df = xp.dot(ytopk.sum(1).T, x)
    DEBUG(7)
    return f / k, df / k


def rcsls(X_src, Y_tgt, Z_src, Z_tgt, R, knn=10):
    DEBUG(1)
    xp = get_array_module(X_src, Y_tgt, Z_src, Z_tgt)
    X_trans = xp.dot(X_src, R.T)
    f = 2 * xp.sum(X_trans * Y_tgt)
    df = 2 * xp.dot(Y_tgt.T, X_src)
    DEBUG(2)
    fk0, dfk0 = getknn(xp.dot(X_trans, Z_tgt.T), X_src, Z_tgt, knn)
    fk1, dfk1 = getknn(xp.dot(xp.dot(Z_src, R.T), Y_tgt.T).T, Y_tgt, Z_src, knn)
    DEBUG(8)
    # f = f - fk0 -fk1
    # df = df - dfk0 - dfk1.T
    return -f, -df


def inspect_matrix(X):
    u, s, vt = xp.linalg.svd(X)
    logging.debug('Squared F-norm: {0:.4f}'.format(float((X**2).sum())))
    logging.debug('Spectral norm: {0:.4f}'.format(float(s[0])))
    logging.debug('10th maximum singular value: {0:.4f}'.format(float(s[10])))
    logging.debug('mean singular value: {0:.4f}'.format(float(s.mean())))


def main(args):
    logging.info(str(args))

    if args.pickle:
        with open(args.source_embedding, 'rb') as fin:
            src_wv = pickle.load(fin)
        with open(args.target_embedding, 'rb') as fin:
            trg_wv = pickle.load(fin)
    else:
        src_wv = WordVecs(args.source_embedding, emb_format=args.format).normalize(args.normalize)
        trg_wv = WordVecs(args.target_embedding, emb_format=args.format).normalize(args.normalize)

    pad_id = src_wv.add_word('<pad>', np.zeros(args.vector_dim, dtype=np.float32))
    src_ds = SentimentDataset(args.source_dataset).to_index(src_wv, binary=(args.loss != 10)).pad(pad_id)
    xsenti = xp.array(src_ds.train[0])
    ysenti = xp.array(src_ds.train[1])
    lsenti = xp.array(src_ds.train[2])

    gold_dict = xp.array(BilingualDict(args.gold_dictionary).get_indexed_dictionary(src_wv, trg_wv), dtype=xp.int32)
    keep_prob = args.dropout_init
    alpha = max(args.alpha, args.alpha_init) if args.alpha_dec else min(args.alpha, args.alpha_init)
    threshold = min(args.threshold, args.threshold_init)
    init_dict = get_unsupervised_init_dict(src_wv.embedding, trg_wv.embedding, args.vocab_cutoff, args.csls, args.normalize, args.direction)
    init_dict = xp.array(init_dict)
    I = xp.identity(args.vector_dim, dtype=xp.float32)

    logging.info('gold dict shape' + str(gold_dict.shape))

    if args.load is not None:
        dic = load_model(args.load)
        W_src = xp.array(dic['W_source'], dtype=xp.float32)
        W_trg = xp.array(dic['W_target'], dtype=xp.float32)
    else:
        W_src = xp.identity(args.vector_dim, dtype=xp.float32)
        W_trg = xp.identity(args.vector_dim, dtype=xp.float32)

    if args.loss >= 6:
        u = xp.zeros(args.vector_dim, dtype=xp.float32)
        v = xp.zeros(args.vector_dim, dtype=xp.float32)
        b = xp.zeros(())
        p = xp.zeros(())
        C = args.C

    src_val_ind = xp.array(np.union1d(asnumpy(gold_dict[:, 0]), asnumpy(xsenti)))
    bdi_obj = BDI(src_wv.embedding, trg_wv.embedding, batch_size=args.batch_size, cutoff_size=args.vocab_cutoff, cutoff_type='both',
                  direction=args.direction, csls=args.csls, batch_size_val=args.val_batch_size, scorer='dot',
                  src_val_ind=src_val_ind, trg_val_ind=gold_dict[:, 1])
    bdi_obj.project(W_src, 'forward', unit_norm=args.normalize_projection)
    bdi_obj.project(W_trg, 'backward', unit_norm=args.normalize_projection)
    curr_dict = init_dict if args.load is None else bdi_obj.get_bilingual_dict_with_cutoff(keep_prob=keep_prob)

    if not args.no_proj_error:
        proj_error = xp.sum((bdi_obj.src_proj_emb[gold_dict[:, 0]] - bdi_obj.trg_proj_emb[gold_dict[:, 1]])**2)
        logging.info('proj error: %.4f' % proj_error)
    # self learning
    try:
        for epoch in range(1, args.epochs):
            logging.debug('running epoch %d...' % epoch)
            logging.debug('alpha: %.4f' % alpha)
            logging.debug('threshold: %.4f' % threshold)

            if epoch % 2 == 0:
                if args.loss == 0:
                    m = args.senti_nsample
                    lr = args.learning_rate
                    X_src = bdi_obj.src_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_proj_emb[curr_dict[:, 1]]
                    ssrc = xp.random.randint(0, xsenti.shape[0], m)
                    strg = xp.random.randint(0, xsenti.shape[0], m)
                    if args.sample_type == 'full':
                        mask = (ysenti[ssrc] == ysenti[strg]).astype(xp.float32) * 2 - 1
                    elif args.sample_type == 'same':
                        mask = (ysenti[ssrc] == ysenti[strg]).astype(xp.float32)
                    elif args.sample_type == 'pos-neg':
                        mask = (ysenti[ssrc] == ysenti[strg]).astype(xp.float32) - 1

                    U_src = bdi_obj.src_emb[xsenti[ssrc]].sum(axis=1) / lsenti[ssrc][:, xp.newaxis]
                    U_trg = bdi_obj.src_proj_emb[xsenti[strg]].sum(axis=1) / lsenti[strg][:, xp.newaxis]
                    U_src *= mask[:, xp.newaxis]
                    logging.debug('number of samples: {0:d}'.format(U_src.shape[0]))
                    prev_loss, loss = float('inf'), float('inf')
                    while prev_loss - loss > 0.05 or loss == float('inf'):
                        prev_W = W_src.copy()
                        grad = -2 * X_src.T.dot(X_trg) - (alpha / m) * U_src.T.dot(U_trg)
                        W_src -= lr * grad
                        W_src = proj_spectral(W_src, threshold=threshold)
                        prev_loss = loss
                        loss = -2 * (X_src.dot(W_src) * X_trg).sum() - (alpha / m) * (U_src.dot(W_src) * U_trg).sum()
                        logging.debug('loss: {0:.4f}'.format(float(loss)))
                    if loss > prev_loss:
                        W_src = prev_W

                elif args.loss == 1:
                    m = args.senti_nsample
                    lr = args.learning_rate
                    X_src = bdi_obj.src_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_proj_emb[curr_dict[:, 1]]
                    ssrc = xp.random.randint(0, xsenti.shape[0], m)
                    strg = xp.random.randint(0, xsenti.shape[0], m)
                    mask = ysenti[ssrc] == ysenti[strg]
                    ssrc = ssrc[mask]
                    strg = strg[mask]
                    U_src = bdi_obj.src_emb[xsenti[ssrc]].sum(axis=1) / lsenti[ssrc][:, xp.newaxis]
                    U_trg = bdi_obj.src_emb[xsenti[strg]].sum(axis=1) / lsenti[strg][:, xp.newaxis]
                    Z = U_src - U_trg
                    logging.debug('number of samples: {0:d}'.format(Z.shape[0]))
                    prev_loss, loss = float('inf'), float('inf')
                    W_src = proj_spectral(xp.linalg.pinv(2 * X_src.T.dot(X_src) + (2 * alpha / m) * Z.T.dot(Z)).dot(X_src.T.dot(X_trg)))
                    W_src = W_src.astype(xp.float32)
                    while lr > 0.0005:
                        prev_W = W_src.copy()
                        prev_loss = loss
                        grad = 2 * ((X_src.T.dot(X_src) + (alpha / m) * Z.T.dot(Z)).dot(W_src) - X_src.T.dot(X_trg))
                        W_src -= lr * grad
                        W_src = proj_spectral(W_src, threshold=threshold)
                        loss = xp.linalg.norm(X_src.dot(W_src) - X_trg)**2 + (alpha / m) * xp.linalg.norm(Z.dot(W_src))**2
                        if loss > prev_loss:
                            lr /= 2
                            W_src = prev_W
                            loss = prev_loss
                        elif prev_loss - loss < 0.5:
                            break
                        logging.debug('loss: {0:.4f}'.format(float(loss)))
                    if loss > prev_loss:
                        W_src = prev_W

                elif args.loss == 2:
                    m = args.senti_nsample
                    lr = args.learning_rate
                    X_src = bdi_obj.src_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_proj_emb[curr_dict[:, 1]]
                    ssrc = xp.random.randint(0, xsenti.shape[0], m)
                    strg = xp.random.randint(0, xsenti.shape[0], m)
                    mask = ysenti[ssrc] == ysenti[strg]
                    ssrc = ssrc[mask]
                    strg = strg[mask]
                    U_src = bdi_obj.src_emb[xsenti[ssrc]].sum(axis=1) / lsenti[ssrc][:, xp.newaxis]
                    U_trg = bdi_obj.src_emb[xsenti[strg]].sum(axis=1) / lsenti[strg][:, xp.newaxis]
                    Z = U_src - U_trg
                    logging.debug('number of samples: {0:d}'.format(Z.shape[0]))
                    prev_loss, loss = float('inf'), float('inf')
                    W_src = proj_spectral(xp.linalg.pinv((alpha / m) * Z.T.dot(Z)).dot(X_src.T.dot(X_trg)))
                    W_src = W_src.astype(xp.float32)
                    while lr > 0.000005:
                        prev_W = W_src.copy()
                        prev_loss = loss
                        grad = -2 * X_src.T.dot(X_trg) + (2 * alpha / m) * Z.T.dot(Z).dot(W_src)
                        W_src -= lr * grad
                        W_src = proj_spectral(W_src, threshold=threshold)
                        loss = -2 * (X_src.dot(W_src) * X_trg).sum() + (alpha / m) * xp.linalg.norm(Z.dot(W_src))**2
                        if loss > prev_loss:
                            lr /= 2
                            W_src = prev_W
                            loss = prev_loss
                        elif prev_loss - loss < 0.5:
                            break
                        logging.debug('loss: {0:.4f}'.format(float(loss)))
                    if loss > prev_loss:
                        W_src = prev_W

                elif args.loss == 3:
                    m = args.senti_nsample
                    X_src = bdi_obj.src_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_proj_emb[curr_dict[:, 1]]
                    ssrc = xp.random.randint(0, xsenti.shape[0], m)
                    strg = xp.random.randint(0, xsenti.shape[0], m)
                    mask = ysenti[ssrc] == ysenti[strg]
                    ssrc = ssrc[mask]
                    strg = strg[mask]
                    U_src = bdi_obj.src_emb[xsenti[ssrc]].sum(axis=1) / lsenti[ssrc][:, xp.newaxis]
                    U_trg = bdi_obj.src_emb[xsenti[strg]].sum(axis=1) / lsenti[strg][:, xp.newaxis]
                    Z = U_src - U_trg
                    logging.debug('number of samples: {0:d}'.format(Z.shape[0]))
                    W_src = xp.linalg.pinv((2 * alpha / m) * Z.T.dot(Z) + (2 * args.beta) * I).dot(X_src.T.dot(X_trg))
                    W_src = W_src.astype(xp.float32)

                elif args.loss == 4:
                    m = args.senti_nsample
                    lr = args.learning_rate
                    X_src = bdi_obj.src_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_proj_emb[curr_dict[:, 1]]
                    pmean = (bdi_obj.src_proj_emb[xsenti[ysenti == 0]].sum(axis=1) / lsenti[ysenti == 0][:, xp.newaxis]).mean(axis=0)
                    nmean = (bdi_obj.src_proj_emb[xsenti[ysenti == 1]].sum(axis=1) / lsenti[ysenti == 1][:, xp.newaxis]).mean(axis=0)
                    sind = xp.random.randint(0, xsenti.shape[0], m)
                    pind = sind[ysenti[sind] == 0]
                    nind = sind[ysenti[sind] == 1]
                    nmean = -pmean
                    pmean /= xp.linalg.norm(pmean)
                    nmean /= xp.linalg.norm(nmean)
                    # DEBUG(bdi_obj.src_emb[xsenti[pind]])
                    # DEBUG(bdi_obj.src_emb[xsenti[nind]])
                    xpos = bdi_obj.src_emb[xsenti[pind]].sum(axis=1) / lsenti[pind, xp.newaxis]
                    xneg = bdi_obj.src_emb[xsenti[nind]].sum(axis=1) / lsenti[nind, xp.newaxis]
                    # DEBUG(xpos.shape)
                    # DEBUG(xneg.shape)
                    DEBUG(xp.linalg.norm(xpos, axis=1))
                    DEBUG(xp.linalg.norm(xneg, axis=1))
                    prev_loss, loss = float('inf'), float('inf')
                    while lr > 0.0000000005:
                        prev_W = W_src.copy()
                        prev_loss = loss
                        grad = -2 * X_src.T.dot(X_trg) - (alpha / m) * (xpos.T.dot(xp.tile(pmean, (xpos.shape[0], 1))) +
                                                                        xneg.T.dot(xp.tile(nmean, (xneg.shape[0], 1))) -
                                                                        xpos.T.dot(xp.tile(nmean, (xpos.shape[0], 1))) -
                                                                        xneg.T.dot(xp.tile(pmean, (xneg.shape[0], 1))))
                        W_src -= lr * grad
                        W_src = proj_spectral(W_src, threshold=threshold)
                        loss = -2 * (X_src.dot(W_src) * X_trg).sum() - (alpha / m) * ((xpos.dot(W_src) * pmean).sum() +
                                                                                      (xneg.dot(W_src) * nmean).sum() -
                                                                                      (xpos.dot(W_src) * nmean).sum() -
                                                                                      (xneg.dot(W_src) * pmean).sum())
                        if loss > prev_loss:
                            lr /= 2
                            W_src = prev_W
                            loss = prev_loss
                        elif prev_loss - loss < 0.5:
                            break
                        logging.debug('loss: {0:.4f}'.format(float(loss)))

                elif args.loss == 5:
                    m = args.senti_nsample
                    lr = args.learning_rate
                    X_src = bdi_obj.src_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_proj_emb[curr_dict[:, 1]]
                    pmean = (bdi_obj.src_proj_emb[xsenti[ysenti == 0]].sum(axis=1) / lsenti[ysenti == 0][:, xp.newaxis]).mean(axis=0)
                    nmean = (bdi_obj.src_proj_emb[xsenti[ysenti == 1]].sum(axis=1) / lsenti[ysenti == 1][:, xp.newaxis]).mean(axis=0)
                    sind = xp.random.randint(0, xsenti.shape[0], m)
                    pind = sind[ysenti[sind] == 0]
                    nind = sind[ysenti[sind] == 1]
                    nmean = -pmean
                    pmean /= xp.linalg.norm(pmean)
                    nmean /= xp.linalg.norm(nmean)
                    # DEBUG(bdi_obj.src_emb[xsenti[pind]])
                    # DEBUG(bdi_obj.src_emb[xsenti[nind]])
                    xpos = bdi_obj.src_emb[xsenti[pind]].sum(axis=1) / lsenti[pind, xp.newaxis]
                    xneg = bdi_obj.src_emb[xsenti[nind]].sum(axis=1) / lsenti[nind, xp.newaxis]
                    length_normalize(xpos, inplace=True)
                    length_normalize(xneg, inplace=True)
                    # DEBUG(xpos.shape)
                    # DEBUG(xneg.shape)
                    DEBUG(xp.linalg.norm(xpos, axis=1))
                    DEBUG(xp.linalg.norm(xneg, axis=1))
                    prev_loss, loss = float('inf'), float('inf')
                    while lr > 0.0000000005:
                        prev_W = W_src.copy()
                        prev_loss = loss
                        grad = 2 * (X_src.T.dot(X_src).dot(W_src) - X_src.T.dot(X_trg)) - (alpha / m) * (xpos.T.dot(xp.tile(pmean, (xpos.shape[0], 1))) +
                                                                                                         xneg.T.dot(xp.tile(nmean, (xneg.shape[0], 1))) -
                                                                                                         xpos.T.dot(xp.tile(nmean, (xpos.shape[0], 1))) -
                                                                                                         xneg.T.dot(xp.tile(pmean, (xneg.shape[0], 1))))
                        W_src -= lr * grad
                        W_src = proj_spectral(W_src, threshold=threshold)
                        loss = xp.linalg.norm(X_src.dot(W_src) - X_trg)**2 - (alpha / m) * ((xpos.dot(W_src) * pmean).sum() +
                                                                                            (xneg.dot(W_src) * nmean).sum() -
                                                                                            (xpos.dot(W_src) * nmean).sum() -
                                                                                            (xneg.dot(W_src) * pmean).sum())
                        if loss > prev_loss:
                            lr /= 2
                            W_src = prev_W
                            loss = prev_loss
                        elif prev_loss - loss < 0.5:
                            break
                        logging.debug('loss: {0:.4f}'.format(float(loss)))
                    logging.debug('euclidean distance between pmean and nmean: %.4f' % xp.linalg.norm(pmean - nmean))
                    logging.debug('dot product of pmean and nmean: %.4f' % pmean.dot(nmean))
                    logging.debug('cosine sim between pmean and nmean: %.4f' % (pmean.dot(nmean) / xp.linalg.norm(pmean) / xp.linalg.norm(nmean)))
                    logging.debug('l2-norm of pmean %.4f' % xp.linalg.norm(pmean))
                    logging.debug('l2-norm of nmean %.4f' % xp.linalg.norm(nmean))

                elif args.loss == 6:
                    m = args.senti_nsample
                    lr = args.learning_rate
                    X_src = bdi_obj.src_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_proj_emb[curr_dict[:, 1]]
                    sind = xp.random.randint(0, xsenti.shape[0], m)
                    Xs = bdi_obj.src_emb[xsenti[sind]].sum(axis=1) / lsenti[sind, xp.newaxis]
                    ys = ysenti[sind] * (-2) + 1  # 1 = positive, -1 = negative
                    DEBUG(ys)
                    DEBUG(bdi_obj.src_emb[xsenti[sind]].shape)
                    DEBUG(bdi_obj.src_emb[xsenti[sind]].sum(axis=1).shape)
                    DEBUG(Xs.shape)
                    DEBUG(sind)
                    # loss = -2 * (X_src.dot(W_src) * X_trg).sum() + (alpha / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(u)) + b) * ys).sum() + C * xp.linalg.norm(W_src)**2
                    loss = (alpha / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(u)) + b) * ys).sum()

                    logging.debug('loss: {0:.4f}'.format(float(loss)))
                    while lr > 0.00000000005:
                        prev_W = W_src.copy()
                        prev_u = u.copy()
                        prev_b = b.copy()
                        prev_loss = loss

                        xtmp = (Xs.dot(W_src.dot(u)) + b) * ys
                        mask = (xtmp < 1).astype(xp.float32)  # 1 = activated, 0 = not activated

                        Zs = Xs * (-ys * mask)[:, xp.newaxis]
                        dW = (alpha / m) * Zs.T.dot(xp.tile(u, (m, 1)))
                        # dW = -2 * X_src.T.dot(X_trg) + (alpha / m) * Zs.T.dot(xp.tile(u, (m, 1))) + (2 * C) * W_src
                        du = (alpha / m) * W_src.T.dot(Zs.sum(axis=0))
                        db = (alpha / m) * (-ys * mask).sum()

                        W_src -= lr * dW
                        u -= lr * du
                        b -= lr * db
                        # W_src = proj_spectral(W_src, threshold=threshold)
                        W_src = proj_spectral(W_src, threshold=args.threshold)

                        loss = (alpha / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(u)) + b) * ys).sum()
                        # loss = -2 * (X_src.dot(W_src) * X_trg).sum() + (alpha / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(u)) + b) * ys).sum() + C * xp.linalg.norm(W_src)**2

                        if loss > prev_loss:
                            lr /= 2
                            W_src, u, b = prev_W, prev_u, prev_b
                            loss = prev_loss
                        elif prev_loss - loss < 0.5:
                            break
                        logging.debug('loss: {0:.4f}'.format(float(loss)))
                    logging.debug('activated number %d' % int(mask.sum()))

                elif args.loss == 7:
                    m = args.senti_nsample
                    lr = args.learning_rate
                    X_src = bdi_obj.src_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_proj_emb[curr_dict[:, 1]]
                    sind = xp.random.randint(0, xsenti.shape[0], m)
                    Xs = bdi_obj.src_emb[xsenti[sind]].sum(axis=1) / lsenti[sind, xp.newaxis]
                    ys = ysenti[sind] * (-2) + 1  # 1 = positive, -1 = negative
                    DEBUG(ys)
                    DEBUG(bdi_obj.src_emb[xsenti[sind]].shape)
                    DEBUG(bdi_obj.src_emb[xsenti[sind]].sum(axis=1).shape)
                    DEBUG(Xs.shape)
                    DEBUG(sind)
                    loss = xp.linalg.norm(X_src.dot(W_src) - X_trg)**2 + (alpha / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(u - v)) + b) * ys).sum()
                    logging.debug('loss: {0:.4f}'.format(float(loss)))
                    while lr > 0.00000005:
                        prev_W = W_src.copy()
                        prev_u = u.copy()
                        prev_b = b.copy()
                        prev_loss = loss

                        xtmp = (Xs.dot(W_src.dot(u)) + b) * ys
                        mask = (xtmp < 1).astype(xp.float32)  # 1 = activated, 0 = not activated

                        Zs = Xs * (-ys * mask)[:, xp.newaxis]
                        dW = 2 * (X_src.T.dot(X_src).dot(W_src) - X_src.T.dot(X_trg)) + (alpha / m) * Zs.T.dot(xp.tile(u, (m, 1)))
                        du = (alpha / m) * W_src.T.dot(Zs.sum(axis=0))
                        db = (alpha / m) * (-ys * mask).sum()

                        W_src -= lr * dW
                        u -= lr * du
                        b -= lr * db
                        W_src = proj_spectral(W_src, threshold=threshold)

                        loss = xp.linalg.norm(X_src.dot(W_src) - X_trg)**2 + (alpha / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(u)) + b) * ys).sum()

                        if loss > prev_loss:
                            lr /= 2
                            W_src, u, b = prev_W, prev_u, prev_b
                            loss = prev_loss
                        elif prev_loss - loss < 5:
                            break
                        logging.debug('loss: {0:.4f}'.format(float(loss)))
                    logging.debug('activated number %d' % int(mask.sum()))

                elif args.loss == 8:
                    m = args.senti_nsample
                    lr = args.learning_rate
                    X_src = bdi_obj.src_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_proj_emb[curr_dict[:, 1]]
                    sind = xp.random.randint(0, xsenti.shape[0], m)
                    Xs = bdi_obj.src_emb[xsenti[sind]].sum(axis=1) / lsenti[sind, xp.newaxis]
                    ys = ysenti[sind] * (-2) + 1  # 1 = positive, -1 = negative
                    # loss = -2 * (X_src.dot(W_src) * X_trg).sum() + (alpha / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(u)) + b) * ys).sum() + C * xp.linalg.norm(W_src)**2
                    loss = (alpha / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(u)) + b) * ys).sum()

                    logging.debug('loss: {0:.4f}'.format(float(loss)))
                    cnt = 0
                    while lr > 0.000000005:
                        prev_W = W_src.copy()
                        prev_u = u.copy()
                        prev_b = b.copy()
                        prev_loss = loss

                        xtmp = (Xs.dot(W_src.dot(u)) + b) * ys
                        mask = (xtmp < 1).astype(xp.float32)  # 1 = activated, 0 = not activated

                        Zs = Xs * (-ys * mask)[:, xp.newaxis]
                        dW = (alpha / m) * Zs.T.dot(xp.tile(u, (m, 1)))
                        # dW = -2 * X_src.T.dot(X_trg) + (alpha / m) * Zs.T.dot(xp.tile(u, (m, 1))) + (2 * C) * W_src
                        du = (alpha / m) * W_src.T.dot(Zs.sum(axis=0))
                        db = (alpha / m) * (-ys * mask).sum()
                        W_src -= lr * dW
                        u -= lr * du
                        b -= lr * db
                        W_src = proj_spectral(W_src, threshold=args.threshold)

                        loss = (alpha / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(u)) + b) * ys).sum()
                        logging.debug('loss: {0:.4f}'.format(float(loss)))
                        # loss = -2 * (X_src.dot(W_src) * X_trg).sum() + (alpha / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(u)) + b) * ys).sum() + C * xp.linalg.norm(W_src)**2

                        if loss >= prev_loss:
                            lr /= 2
                            W_src, u, b = prev_W, prev_u, prev_b
                            loss = prev_loss
                        else:
                            cnt += 1
                            if cnt == 10:
                                break
                    logging.debug('activated number %d' % int(mask.sum()))

                elif args.loss == 9:
                    m = args.senti_nsample
                    lr = args.learning_rate
                    X_src = bdi_obj.src_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_proj_emb[curr_dict[:, 1]]
                    sind = xp.random.randint(0, xsenti.shape[0], m)
                    Xs = bdi_obj.src_emb[xsenti[sind]].sum(axis=1) / lsenti[sind, xp.newaxis]
                    ys = ysenti[sind] * (-2) + 1  # 1 = positive, -1 = negative
                    # loss = -2 * (X_src.dot(W_src) * X_trg).sum() + (alpha / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(u)) + b) * ys).sum() + C * xp.linalg.norm(W_src)**2
                    loss = (alpha / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(u)) + b) * ys).sum()

                    logging.debug('loss: {0:.4f}'.format(float(loss)))
                    cnt = 0
                    while lr > 0.0000000000000005:
                        prev_W = W_src.copy()
                        prev_u = u.copy()
                        prev_b = b.copy()
                        prev_loss = loss

                        xtmp = (Xs.dot(W_src.dot(u)) + b) * ys
                        mask = (xtmp < 1).astype(xp.float32)  # 1 = activated, 0 = not activated

                        Zs = Xs * (-ys * mask)[:, xp.newaxis]
                        dW = (alpha / m) * Zs.T.dot(xp.tile(u, (m, 1)))
                        # dW = -2 * X_src.T.dot(X_trg) + (alpha / m) * Zs.T.dot(xp.tile(u, (m, 1))) + (2 * C) * W_src
                        du = (alpha / m) * W_src.T.dot(Zs.sum(axis=0))
                        db = (alpha / m) * (-ys * mask).sum()
                        W_src -= lr * dW
                        u -= lr * du
                        b -= lr * db
                        W_src = proj_spectral(W_src, threshold=args.threshold)

                        loss = (alpha / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(u)) + b) * ys).sum()
                        logging.debug('loss: {0:.4f}'.format(float(loss)))
                        # loss = -2 * (X_src.dot(W_src) * X_trg).sum() + (alpha / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(u)) + b) * ys).sum() + C * xp.linalg.norm(W_src)**2

                        if loss >= prev_loss:
                            lr /= 2
                            W_src, u, b = prev_W, prev_u, prev_b
                            loss = prev_loss
                        else:
                            cnt += 1
                            if cnt == 10:
                                break
                    logging.debug('activated number %d' % int(mask.sum()))

                elif args.loss == 10:
                    m = args.senti_nsample
                    lr = args.learning_rate
                    X_src = bdi_obj.src_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_proj_emb[curr_dict[:, 1]]
                    sind = xp.random.randint(0, xsenti.shape[0], m)
                    Xs = bdi_obj.src_emb[xsenti[sind]].sum(axis=1) / lsenti[sind, xp.newaxis]
                    # DEBUG(ysenti[sind])
                    ys = (ysenti[sind] >= 2).astype(xp.float32) * (-2) + 1  # 1 = positive, -1 = negative
                    ts = ((ysenti[sind] == 1) | (ysenti[sind] == 3)).astype(xp.float32) + 1  # 1 = pos/neg, 2 = strpos/strneg
                    # DEBUG(ys)
                    # DEBUG(ts)
                    # loss = -2 * (X_src.dot(W_src) * X_trg).sum() + (alpha / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(u)) + b) * ys).sum() + C * xp.linalg.norm(W_src)**2
                    loss = (alpha / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(u)) + b) * ys).dot(ts).sum()

                    logging.debug('loss: {0:.4f}'.format(float(loss)))
                    cnt = 0
                    while lr > 0.0000000000000005:
                        prev_W = W_src.copy()
                        prev_u = u.copy()
                        prev_b = b.copy()
                        prev_loss = loss

                        xtmp = (Xs.dot(W_src.dot(u)) + b) * ys
                        mask = (xtmp < 1).astype(xp.float32)  # 1 = activated, 0 = not activated

                        Zs = Xs * (-ys * mask)[:, xp.newaxis]
                        dW = (alpha / m) * (Zs.T * ts).dot(xp.tile(u, (m, 1)))
                        # dW = -2 * X_src.T.dot(X_trg) + (alpha / m) * Zs.T.dot(xp.tile(u, (m, 1))) + (2 * C) * W_src
                        du = (alpha / m) * W_src.T.dot((Zs * ts[:, xp.newaxis]).sum(axis=0))
                        db = (alpha / m) * (-ys * mask).dot(ts).sum()
                        W_src -= lr * dW
                        u -= lr * du
                        b -= lr * db
                        W_src = proj_spectral(W_src, threshold=threshold)

                        loss = (alpha / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(u)) + b) * ys).dot(ts).sum()
                        logging.debug('loss: {0:.4f}'.format(float(loss)))
                        # loss = -2 * (X_src.dot(W_src) * X_trg).sum() + (alpha / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(u)) + b) * ys).sum() + C * xp.linalg.norm(W_src)**2

                        if loss >= prev_loss:
                            lr /= 2
                            W_src, u, b = prev_W, prev_u, prev_b
                            loss = prev_loss
                        else:
                            cnt += 1
                            if cnt == 10:
                                break
                    logging.debug('activated number %d' % int(mask.sum()))

                elif args.loss == 11:
                    m = args.senti_nsample
                    lr = args.learning_rate
                    X_src = bdi_obj.src_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_proj_emb[curr_dict[:, 1]]
                    sind = xp.random.randint(0, xsenti.shape[0], m)
                    Xs = bdi_obj.src_emb[xsenti[sind]].sum(axis=1) / lsenti[sind, xp.newaxis]
                    # DEBUG(ysenti[sind])
                    ys = (ysenti[sind] >= 2).astype(xp.float32) * (-2) + 1  # 1 = positive, -1 = negative
                    ts = ((ysenti[sind] == 1) | (ysenti[sind] == 3)).astype(xp.float32) + 1  # 1 = pos/neg, 2 = strpos/strneg
                    # DEBUG(ys)
                    # DEBUG(ts)
                    # loss = -2 * (X_src.dot(W_src) * X_trg).sum() + (alpha / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(u)) + b) * ys).sum() + C * xp.linalg.norm(W_src)**2
                    loss = (alpha / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(u)) + b) * ys).dot(ts).sum()

                    logging.debug('loss: {0:.4f}'.format(float(loss)))
                    cnt = 0
                    while lr > 0.0000000000000005:
                        prev_W = W_src.copy()
                        prev_u = u.copy()
                        prev_b = b.copy()
                        prev_loss = loss

                        xtmp = (Xs.dot(W_src.dot(u)) + b) * ys
                        mask = (xtmp < 1).astype(xp.float32)  # 1 = activated, 0 = not activated

                        Zs = Xs * (-ys * mask)[:, xp.newaxis]
                        dW = (alpha / m) * (Zs.T * ts).dot(xp.tile(u, (m, 1)))
                        # dW = -2 * X_src.T.dot(X_trg) + (alpha / m) * Zs.T.dot(xp.tile(u, (m, 1))) + (2 * C) * W_src
                        du = (alpha / m) * W_src.T.dot((Zs * ts[:, xp.newaxis]).sum(axis=0))
                        db = (alpha / m) * (-ys * mask).dot(ts).sum()
                        W_src -= lr * dW
                        u -= lr * du
                        b -= lr * db
                        W_src = proj_spectral(W_src, threshold=threshold)

                        loss = (alpha / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(u)) + b) * ys).dot(ts).sum()
                        logging.debug('loss: {0:.4f}'.format(float(loss)))
                        # loss = -2 * (X_src.dot(W_src) * X_trg).sum() + (alpha / m) * xp.maximum(0, 1 - (Xs.dot(W_src.dot(u)) + b) * ys).sum() + C * xp.linalg.norm(W_src)**2

                        if loss >= prev_loss:
                            lr /= 2
                            W_src, u, b = prev_W, prev_u, prev_b
                            loss = prev_loss
                        else:
                            cnt += 1
                            if cnt == 10:
                                break
                    logging.debug('activated number %d' % int(mask.sum()))

                # logging.debug('squared f-norm of W_src: %.4f' % xp.sum(W_src**2))
                # logging.debug('spectral norm of W_src: %.4f' % spectral_norm(W_src))
                # logging.debug('spectral_norm')
                inspect_matrix(W_src)
                bdi_obj.project(W_src, 'forward', unit_norm=args.normalize_projection)

            elif epoch % 2 == 1:
                if args.loss == 3:
                    X_src = bdi_obj.src_proj_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_emb[curr_dict[:, 1]]
                    W_trg = (1 / 2 / args.beta) * X_trg.T.dot(X_src)
                    # W_trg = I

                # elif args.loss == 1:
                #     lr = args.learning_rate
                #     X_src = bdi_obj.src_proj_emb[curr_dict[:, 0]]
                #     X_trg = bdi_obj.trg_emb[curr_dict[:, 1]]
                #     prev_loss, loss = float('inf'), float('inf')
                #     while lr > 0.0006:
                #         prev_W = W_trg.copy()
                #         grad = -2 * X_trg.T.dot(X_src)
                #         W_trg -= lr * grad
                #         W_trg = proj_spectral(W_trg, threshold=threshold)
                #         prev_loss = loss
                #         loss = -2 * (X_trg.dot(W_trg) * X_src).sum()
                #         if loss > prev_loss:
                #             lr /= 2
                #             W_trg = prev_W
                #             loss = prev_loss
                #         elif prev_loss - loss < 0.5:
                #             break
                #         logging.debug('loss: {0:.4f}'.format(float(loss)))
                #     if loss > prev_loss:
                #         W_trg = prev_W

                elif args.loss == 1:
                    lr = args.learning_rate
                    X_src = bdi_obj.src_proj_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_emb[curr_dict[:, 1]]
                    prev_loss, loss = float('inf'), float('inf')
                    while lr > 0.0006:
                        prev_W = W_trg.copy()
                        grad = 2 * (X_trg.T.dot(X_trg).dot(W_trg) - X_trg.T.dot(X_src))
                        W_trg -= lr * grad
                        W_trg = proj_spectral(W_trg, threshold=threshold)
                        prev_loss = loss
                        loss = xp.linalg.norm(X_trg.dot(W_trg) - X_src)**2
                        if loss > prev_loss:
                            lr /= 2
                            W_trg = prev_W
                            loss = prev_loss
                        elif prev_loss - loss < 0.5:
                            break
                        logging.debug('loss: {0:.4f}'.format(float(loss)))
                    if loss > prev_loss:
                        W_trg = prev_W

                elif args.loss == 5:
                    lr = args.learning_rate
                    X_src = bdi_obj.src_proj_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_emb[curr_dict[:, 1]]
                    prev_loss, loss = float('inf'), float('inf')
                    while lr > 0.00005:
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
                        elif prev_loss - loss < 0.5:
                            break
                        logging.debug('loss: {0:.4f}'.format(float(loss)))
                    if loss > prev_loss:
                        W_trg = prev_W

                elif args.loss == 7:
                    lr = args.learning_rate
                    X_src = bdi_obj.src_proj_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_emb[curr_dict[:, 1]]
                    prev_loss, loss = float('inf'), float('inf')
                    while lr > 0.0000005:
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
                        elif prev_loss - loss < 5:
                            break
                        logging.debug('loss: {0:.4f}'.format(float(loss)))
                    if loss > prev_loss:
                        W_trg = prev_W

                elif args.loss == 6:
                    lr = args.learning_rate
                    X_src = bdi_obj.src_proj_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_emb[curr_dict[:, 1]]
                    prev_loss, loss = float('inf'), float('inf')
                    while lr > 0.006:
                        prev_W = W_trg.copy()
                        prev_loss = loss
                        grad = -2 * X_trg.T.dot(X_src) + (2 * C) * W_trg
                        W_trg -= lr * grad
                        W_trg = proj_spectral(W_trg, threshold=threshold)
                        loss = -2 * (X_trg.dot(W_trg) * X_src).sum() + C * xp.linalg.norm(W_trg)**2
                        if loss > prev_loss:
                            lr /= 2
                            W_trg = prev_W
                            loss = prev_loss
                        elif prev_loss - loss < 0.5:
                            break
                        logging.debug('loss: {0:.4f}'.format(float(loss)))
                    if loss > prev_loss:
                        W_trg = prev_W

                elif args.loss == 8 or (args.loss in (9, 10) and epoch == 1):
                    lr = args.learning_rate
                    X_src = bdi_obj.src_proj_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_emb[curr_dict[:, 1]]
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
                        logging.debug('loss: {0:.4f}'.format(float(loss)))
                    if loss > prev_loss:
                        W_trg = prev_W

                elif args.loss in (9, 10):
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
                    if loss > prev_loss:
                        W_trg = prev_W

                elif args.loss == 11:
                    lr = args.learning_rate
                    cf = args.vocab_cutoff
                    X_src = bdi_obj.src_proj_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_emb[curr_dict[:, 1]]
                    loss, grad = rcsls(X_trg, X_src, bdi_obj.trg_emb[:cf], bdi_obj.src_proj_emb[:cf], W_trg.T, 10)
                    grad = grad.T
                    logging.debug('loss: {0:.4f}'.format(float(loss)))
                    while lr > 0.0000005:
                        prev_loss = loss
                        prev_W = W_trg.copy()
                        prev_grad = grad.copy()
                        W_trg -= lr * grad
                        W_trg = proj_spectral(W_trg, threshold=threshold)
                        loss, grad = rcsls(X_trg, X_src, bdi_obj.trg_emb[:cf], bdi_obj.src_proj_emb[:cf], W_trg.T, 10)
                        grad = grad.T
                        if loss > prev_loss:
                            lr /= 2
                            W_trg = prev_W
                            loss = prev_loss
                            grad = prev_grad
                        elif prev_loss - loss < 0.5:
                            break
                        logging.debug('loss: {0:.4f}'.format(float(loss)))
                    if loss > prev_loss:
                        W_trg = prev_W

                else:
                    lr = args.learning_rate
                    X_src = bdi_obj.src_proj_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_emb[curr_dict[:, 1]]
                    prev_loss, loss = float('inf'), float('inf')
                    while lr > 0.006:
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
                        logging.debug('loss: {0:.4f}'.format(float(loss)))
                    if loss > prev_loss:
                        W_trg = prev_W

                # logging.debug('squared f-norm of W_trg: %.4f' % xp.sum(W_trg**2))
                # logging.debug('spectral norm of W_trg: %.4f' % spectral_norm(W_trg))
                inspect_matrix(W_trg)
                bdi_obj.project(W_trg, 'backward', unit_norm=args.normalize_projection)

            if not args.no_proj_error:
                proj_error = xp.sum((bdi_obj.src_proj_emb[gold_dict[:, 0]] - bdi_obj.trg_proj_emb[gold_dict[:, 1]])**2)
                logging.info('proj error: %.4f' % proj_error)

            # dictionary induction
            if epoch % 2 == 1:
                curr_dict = bdi_obj.get_bilingual_dict_with_cutoff(keep_prob=keep_prob)

            # update keep_prob
            if (epoch + 1) % (args.dropout_interval * 2) == 0:
                keep_prob = min(1., keep_prob + args.dropout_step)

            # update alpha
            if args.alpha_inc:
                alpha = min(args.alpha_step + alpha, args.alpha)
            elif args.alpha_dec:
                alpha = max(alpha - args.alpha_step, args.alpha)
            elif args.alpha_mul:
                alpha = min(args.alpha_factor * alpha, args.alpha)

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
            # W_src = proj_spectral(W_src, threshold=args.threshold)
            W_trg = proj_spectral(W_trg, threshold=args.threshold)
        if args.loss >= 6:
            save_model(asnumpy(W_src), asnumpy(W_trg), args.source_lang,
                       args.target_lang, args.model, args.save_path,
                       alpha=args.alpha, alpha_init=args.alpha_init, dropout_init=args.dropout_init, u=asnumpy(u), b=asnumpy(b))
        else:
            save_model(asnumpy(W_src), asnumpy(W_trg), args.source_lang,
                       args.target_lang, args.model, args.save_path,
                       alpha=args.alpha, alpha_init=args.alpha_init, dropout_init=args.dropout_init)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], default=0, help='type of loss function')
    parser.add_argument('-C', '--C', type=float, default=0, help='type of loss function')

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
    mapping_group.add_argument('-b', '--beta', type=float, default=0, help='regularization parameter')
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
    alpha_group.add_argument('--alpha_factor', type=float, default=1.01, help='multiply alpha by a factor each epoch')
    alpha_group.add_argument('--alpha_step', type=float, default=0.02, help='multiply alpha by a factor each epoch')
    alpha_update = alpha_group.add_mutually_exclusive_group()
    alpha_update.add_argument('--alpha_inc', action='store_true', help='increase alpha by a step each epoch')
    alpha_update.add_argument('--alpha_dec', action='store_true', help='decrease alpha by a step each epoch')
    alpha_update.add_argument('--alpha_mul', action='store_true', help='multiply alpha by a factor each epoch')

    induction_group = parser.add_argument_group()
    induction_group.add_argument('-vc', '--vocab_cutoff', default=10000, type=int, help='restrict the vocabulary to k most frequent words')
    induction_group.add_argument('--csls', type=int, default=10, help='number of csls neighbours')
    induction_group.add_argument('--dropout_init', type=float, default=0.1, help='initial keep prob of the dropout machanism')
    induction_group.add_argument('--dropout_interval', type=int, default=50, help='increase keep_prob every m steps')
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
                        no_target_senti=True, model='ubise', normalize_projection=True,
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
