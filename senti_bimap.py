import argparse
import pickle
import logging
import sys
import os
import re
import numpy as np
from utils import utils
from utils.cupy_utils import *


def sample_senti_vecs(xpos, xneg, num_sample):
    xp = get_array_module(xpos, xneg)
    return xp.zeros((num_sample, xpos.shape[1]), dtype=xp.float32), xp.zeros((num_sample, xpos.shape[1]), dtype=xp.float32)
    if xp == np:
        xpos = xp.random.permutation(xpos)
        xneg = xp.random.permutation(xneg)
    else:
        xp.random.permutation(xpos)
        xp.random.permutation(xneg)
    nsample = min(xpos.shape[0], xneg.shape[0], num_sample)
    return xpos[:nsample], xneg[:nsample]


def get_pos_neg_vecs(X, y):
    xp = get_array_module(X, y)
    pos_mask = y <= 1
    xpos = X[pos_mask]
    xneg = X[~pos_mask]
    return xpos, xneg


def get_projection_with_senti(X_src, X_trg, pos, neg, alpha, direction='forward', orthogonal=False, normalize=False, spectral=False, threshold=1.):
    xp = get_array_module(X_src, X_trg, pos, neg)
    if orthogonal:
        if direction == 'forward':
            u, s, vt = xp.linalg.svd(xp.dot(X_trg.T, X_src))
            W = xp.dot(vt.T, u.T)
        if direction == 'backward':
            u, s, vt = xp.linalg.svd(xp.dot(X_src.T, X_trg))
            W = xp.dot(vt.T, u.T)
    else:
        if direction == 'forward':
            W = xp.linalg.pinv(X_src.T.dot(X_src) - alpha * (pos - neg).T.dot(pos - neg)).dot(X_src.T.dot(X_trg))
        elif direction == 'backward':
            W = xp.linalg.pinv(X_trg.T.dot(X_trg) - alpha * (pos - neg).T.dot(pos - neg)).dot(X_trg.T.dot(X_src))

        if spectral:
            W = proj_spectral(W, threshold=threshold)
        if normalize:
            fnorm = xp.sqrt(xp.sum(W**2))
            W *= xp.sqrt(W.shape[0]) / fnorm
    return W


def proj_spectral(W, tanh=False, threshold=1.):
    xp = get_array_module(W)
    u, s, vt = xp.linalg.svd(W)
    if tanh:
        s = xp.tanh(s)
    else:
        s[s > threshold] = threshold
        s[s < 0] = 0
    return xp.dot(u, xp.dot(xp.diag(s), vt))


def main(args):
    logging.info(str(args))

    if args.plot:
        raise NotImplementedError  # TODO

    if not os.path.exists('log'):
        os.mkdir('log')
    log_file = open(args.log, 'w', encoding='utf-8')

    src_wv = utils.WordVecs(args.source_embedding).normalize(args.normalize)
    trg_wv = utils.WordVecs(args.target_embedding).normalize(args.normalize)
    src_emb = xp.array(src_wv.embedding, dtype=xp.float32)
    trg_emb = xp.array(trg_wv.embedding, dtype=xp.float32)
    src_ds = utils.SentimentDataset(args.source_dataset).to_index(src_wv).to_vecs(src_wv.embedding)
    trg_ds = utils.SentimentDataset(args.target_dataset).to_index(trg_wv).to_vecs(trg_wv.embedding)
    src_pos, src_neg = get_pos_neg_vecs(xp.array(src_ds.train[0]), xp.array(src_ds.train[1]))
    trg_pos, trg_neg = get_pos_neg_vecs(xp.array(trg_ds.train[0]), xp.array(trg_ds.train[1]))
    gold_dict = xp.array(utils.BilingualDict(args.gold_dictionary).get_indexed_dictionary(src_wv, trg_wv), dtype=xp.int32)
    keep_prob = args.dropout_init

    if args.init_num:
        init_dict = get_numeral_init_dict(src_wv, trg_wv)
    elif args.init_unsupervised:
        init_dict = utils.get_unsupervised_init_dict(src_emb, trg_emb, args.vocab_cutoff, args.csls, args.normalize, args.direction)
    else:
        init_dict = xp.array(utils.BilingualDict(args.init_dictionary).get_indexed_dictionary(src_wv, trg_wv), dtype=xp.int32)
    del src_wv, trg_wv

    if args.load != '':
        with open(args.load, 'rb') as fin:
            W_src, W_trg = pickle.load(fin)
    else:
        W_src = W_trg = xp.identity(args.vector_dim, dtype=xp.float32)

    bdi_obj = utils.BDI(src_emb, trg_emb, batch_size=args.batch_size, cutoff_size=args.vocab_cutoff, cutoff_type='both',
                        direction=args.direction, csls=args.csls, batch_size_val=args.val_batch_size, scorer=args.scorer)
    bdi_obj.project(W_src, 'forward', unit_norm=args.spectral)
    bdi_obj.project(W_trg, 'backward', unit_norm=args.spectral)
    curr_dict = init_dict if args.load == '' else bdi_obj.get_bilingual_dict_with_cutoff(keep_prob=keep_prob)

    # self learning
    for epoch in range(args.epochs):
        # compute projection matrix
        X_src = src_emb[curr_dict[:, 0]]
        X_trg = trg_emb[curr_dict[:, 1]]
        if epoch % 2 == 0:
            X_trg.dot(W_trg, out=X_trg)
            if args.spectral:
                utils.length_normalize(X_trg, inplace=True)
            xpos, xneg = sample_senti_vecs(src_pos, src_neg, args.senti_nsample)
            W_src = get_projection_with_senti(X_src, X_trg, xpos, xneg, args.alpha, 'forward', args.orthogonal, args.normalize_W, args.spectral, args.threshold)
            logging.info('squared f-norm of W_src: %.4f' % xp.sum(W_src**2))
            bdi_obj.project(W_src, 'forward', unit_norm=args.spectral)
        elif epoch % 2 == 1:
            X_src.dot(W_src, out=X_src)
            if args.spectral:
                utils.length_normalize(X_src, inplace=args.spectral)
            xpos, xneg = sample_senti_vecs(trg_pos, trg_neg, args.senti_nsample)
            W_trg = get_projection_with_senti(X_src, X_trg, xpos, xneg, args.alpha, 'backward', args.orthogonal, False, args.spectral, args.threshold)
            logging.info('squared f-norm of W_trg: %.4f' % xp.sum(W_trg**2))
            bdi_obj.project(W_trg, 'backward', unit_norm=args.spectral)

        if args.spectral:
            proj_error = xp.sum((utils.length_normalize(src_emb[gold_dict[:, 0]] @ W_src, False) - utils.length_normalize(trg_emb[gold_dict[:, 1]] @ W_trg, False))**2)
        else:
            proj_error = xp.sum((src_emb[gold_dict[:, 0]] @ W_src - trg_emb[gold_dict[:, 1]] @ W_trg)**2)
        logging.info('proj error: %.4f' % proj_error)

#         if epoch % 2 == 1:
        # dictionary induction
        curr_dict = bdi_obj.get_bilingual_dict_with_cutoff(keep_prob=keep_prob)

        # update keep_prob
        if (epoch + 1) % (args.dropout_interval * 2) == 0:
            keep_prob = min(1., keep_prob + args.dropout_step)

        # valiadation
        if not args.no_valiadation and (epoch + 1) % args.valiadation_step == 0 or epoch == (args.epochs - 1):
            val_trg_ind = bdi_obj.get_target_indices(gold_dict[:, 0])
            accuracy = xp.mean((val_trg_ind == gold_dict[:, 1]).astype(xp.int32))
            logging.info('epoch: %d   accuracy: %.4f   dict_size: %d' % (epoch, accuracy, curr_dict.shape[0]))
            log_file.write('%d,%.4f\n' % (epoch, accuracy))

    log_file.close()

    # save W_trg
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    with open(args.save_path, 'wb') as fout:
        pickle.dump([asnumpy(W_src), asnumpy(W_trg)], fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-se', '--source_embedding', default='./emb/en.bin', help='monolingual word embedding of the source language (default: ./emb/en.bin)')
    parser.add_argument('-te', '--target_embedding', default='./emb/es.bin', help='monolingual word embedding of the target language (default: ./emb/es.bin)')
    parser.add_argument('-sd', '--source_dataset', default='./datasets/en/opener_sents/', help='source sentiment dataset')
    parser.add_argument('-td', '--target_dataset', default='./datasets/es/opener_sents/', help='target sentiment dataset')
    parser.add_argument('-gd', '--gold_dictionary', default='./lexicons/apertium/en-es.txt', help='gold bilingual dictionary for evaluation(default: ./lexicons/apertium/en-es.txt)')
    parser.add_argument('--load', type=str, default='', help='restore W_src and W_trg from a file')
    parser.add_argument('-vd', '--vector_dim', default=300, type=int, help='dimension of each word vector (default: 300)')
    parser.add_argument('-e', '--epochs', default=500, type=int, help='training epochs (default: 500)')
    parser.add_argument('-bs', '--batch_size', default=2000, type=int, help='training batch size (default: 2000)')
    parser.add_argument('-vbs', '--val_batch_size', default=500, type=int, help='training batch size (default: 500)')
    parser.add_argument('--no_valiadation', action='store_true', help='disable valiadation at each iteration')
    parser.add_argument('--valiadation_step', type=int, default=50, help='valiadation frequency')
    parser.add_argument('--debug', action='store_const', dest='loglevel', default=logging.INFO, const=logging.DEBUG, help='print debug info')
    parser.add_argument('--save_path', default='./checkpoints/senti.bin', help='file to save W_src and W_trg')
    parser.add_argument('--cuda', action='store_true', help='use cuda to accelerate')
    parser.add_argument('--log', default='./log/init100.csv', type=str, help='file to print log')
    parser.add_argument('--plot', action='store_true', help='plot results')

    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument('-d', '--init_dictionary', default='./init_dict/init100.txt', help='bilingual dictionary for learning bilingual mapping (default: ./init_dict/init100.txt)')
    init_group.add_argument('--init_num', action='store_true', help='use numerals as initial dictionary')
    init_group.add_argument('--init_unsupervised', action='store_true', help='use unsupervised init')

    mapping_group = parser.add_argument_group()
    mapping_group.add_argument('--normalize', choices=['unit', 'center', 'unitdim', 'centeremb', 'none'], nargs='*', default=['unit', 'center', 'unit'], help='normalization actions')
    mapping_group.add_argument('--orthogonal', action='store_true', help='restrict projection matrix to be orthogonal')
    mapping_group.add_argument('--spectral', action='store_true', help='restrict projection matrix to spectral domain')
    mapping_group.add_argument('--threshold', type=float, default=1., help='thresholding the singular value of W')
    mapping_group.add_argument('--normalize_W', action='store_true', help='add f-norm restriction on W')
    mapping_group.add_argument('-a', '--alpha', type=float, default=0.1, help='trade-off between sentiment and alignment')
    mapping_group.add_argument('-n', '--senti_nsample', type=int, default=200, help='sentiment examples')

    induction_group = parser.add_argument_group()
    induction_group.add_argument('-vc', '--vocab_cutoff', default=10000, type=int, help='restrict the vocabulary to k most frequent words')
    induction_group.add_argument('--csls', type=int, default=10, help='number of csls neighbours')
    induction_group.add_argument('--dropout_init', type=float, default=0.1, help='initial keep prob of the dropout machanism')
    induction_group.add_argument('--dropout_interval', type=int, default=30, help='increase keep_prob every m steps')
    induction_group.add_argument('--dropout_step', type=float, default=0.1, help='increase keep_prob by a small step')
    induction_group.add_argument('--direction', choices=['forward', 'backward', 'union'], default='union', help='direction of dictionary induction')
    induction_group.add_argument('--scorer', choices=['dot', 'cos', 'euclidean'], default='dot', help='scorer for nearest neighbour retrieval')

    recommend_group = parser.add_mutually_exclusive_group()
    recommend_group.add_argument('-u', '--unsupervised', action='store_true', help='use recommended settings')
    recommend_group.add_argument('-s5', '--supervised5000', action='store_true', help='use supervised5000 settings')
    recommend_group.add_argument('-s1', '--supervised100', action='store_true', help='use supervised100 settings')
    recommend_group.add_argument('--senti', action='store_true', help='use unsupervised + senti settings')
    recommend_group.add_argument('--unconstrained', action='store_true', help='use unsupervised + unconstrained settings')

    args = parser.parse_args()
    if args.unsupervised:
        parser.set_defaults(init_unsupervised=True, csls=10, direction='union', cuda=True, normalize=['center', 'unit'],
                            vocab_cutoff=10000, orthogonal=True, log='./log/unsupervised.csv')
    elif args.supervised5000:
        parser.set_defaults(init_dictionary='./init_dict/init5000.txt', csls=10, direction='union', cuda=True,
                            normalize=['center', 'unit'], vocab_cutoff=10000, orthogonal=True, log='./log/supervised5000.csv')
    elif args.supervised100:
        parser.set_defaults(init_dictionary='./init_dict/init100.txt', csls=10, direction='union', cuda=True,
                            normalize=['center', 'unit'], vocab_cutoff=10000, orthogonal=True, log='./log/supervised100.csv')
    elif args.senti:
        parser.set_defaults(init_unsupervised=True, csls=10, direction='union', cuda=True, normalize=['center', 'unit'],
                            vocab_cutoff=10000, alpha=0.1, senti_nsample=200, log='./log/senti.csv', spectral=True, threshold=1.)
    elif args.unconstrained:
        parser.set_defaults(init_unsupervised=True, csls=10, direction='union', cuda=True, normalize=['center', 'unit'],
                            vocab_cutoff=10000, alpha=0.1, senti_nsample=200, log='./log/senti.csv', scorer='euclidean')

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
