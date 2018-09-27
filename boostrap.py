import argparse
import pickle
import logging
import sys
import os
import re
import numpy as np
from utils import utils
from utils.cupy_utils import *


def plot(files, labels):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    acc = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as fin:
            acc.append([float(line.split(',')[1]) for line in fin])
    for y, l in zip(acc, labels):
        ax.plot(range(1, 21), y, label=l)
    ax.legend(loc='lower right')
    ax.set_xlim(1, 20)
    ax.set_xlabel('iterations')
    ax.set_ylabel('accuracy')


def top_k_mean(X, k, inplace=False):
    size = X.shape[0]
    ans = xp.zeros(size, dtype=xp.float32)
    if k == 0:
        return ans
    if not inplace:
        X = X.copy()
    min_val = X.min()
    ind0 = xp.arange(size)
    ind1 = xp.zeros(size, dtype=xp.int32)
    for i in range(k):
        xp.argmax(X, axis=1, out=ind1)
        ans += X[ind0, ind1]
        X[ind0, ind1] = min_val
    ans /= k
    return ans


def dropout(X, keep_prob, inplace=True):
    mask = xp.random.rand(*X.shape) < keep_prob
    if inplace:
        X *= mask
    else:
        X = X * mask
    return X


def get_unsupervised_init_dict(src_emb, trg_emb, vocab_size, num_neighbours, norm_actions, direction):
    sim_size = min(src_emb.shape[0], trg_emb.shape[0], vocab_size) if vocab_size > 0 else min(src_emb.shape[0], trg_emb.shape[0])
    u, s, vt = xp.linalg.svd(src_emb[:sim_size], full_matrices=False)
    src_sim = (u * s) @ u.T
    u, s, vt = xp.linalg.svd(trg_emb[:sim_size], full_matrices=False)
    trg_sim = (u * s) @ u.T
    del u, s, vt

    src_sim.sort(axis=1)
    trg_sim.sort(axis=1)
    utils.normalize(src_sim, norm_actions)
    utils.normalize(trg_sim, norm_actions)
    sim = xp.dot(src_sim, trg_sim.T)
    del src_sim, trg_sim
    src_knn_sim = top_k_mean(sim, num_neighbours)
    trg_knn_sim = top_k_mean(sim.T, num_neighbours)
    sim -= src_knn_sim[:, xp.newaxis] / 2 + trg_knn_sim / 2

    if args.direction == 'forward':
        init_dict = xp.stack([xp.arange(sim_size), xp.argmax(sim, axis=1)], axis=1)
    elif args.direction == 'backward':
        init_dict = xp.stack([xp.argmax(sim, axis=0), xp.arange(sim_size)], axis=1)
    elif args.direction == 'union':
        init_dict = xp.stack([xp.concatenate((xp.arange(sim_size), xp.argmax(sim, axis=0))), xp.concatenate((xp.argmax(sim, axis=1), xp.arange(sim_size)))], axis=1)
    return init_dict


def get_numeral_init_dict(src_wv, trg_wv):
    num_regex = re.compile('^[0-9]+$')
    src_nums = {w for w in src_wv.vocab if num_regex.match(w) is not None}
    trg_nums = {w for w in trg_wv.vocab if num_regex.match(w) is not None}
    common = src_nums & trg_nums
    init_dict = xp.array([[src_wv.word2index(w), trg_wv.word2index(w)] for w in common], dtype=xp.int32)
    return init_dict


def get_W_target(X_src, X_trg, orthogonal, out=None):
    if orthogonal:
        u, s, vt = xp.linalg.svd(xp.dot(X_src.T, X_trg))
        W_trg = xp.dot(vt.T, u.T, out=out)
    else:
        W_trg = xp.dot(xp.linalg.pinv(X_trg), X_src, out=out)
    if out is None:
        return W_trg


def main(args):
    logging.info(str(args))

    if args.plot:
        labels = (25, 50, 100, 200, 500, 1000, 2000, 5000)
        plot(['log/init%d.csv' % t for t in labels], labels)
        plt.show()
        sys.exit(0)

    if not os.path.exists('log'):
        os.mkdir('log')
    log_file = open(args.log, 'w', encoding='utf-8')


    src_wv = utils.WordVecs(args.source_embedding).normalize(args.normalize)
    trg_wv = utils.WordVecs(args.target_embedding).normalize(args.normalize)
    src_emb = xp.array(src_wv.embedding, dtype=xp.float32)
    trg_emb = xp.array(trg_wv.embedding, dtype=xp.float32)
    gold_dict = xp.array(utils.BilingualDict(args.gold_dictionary).get_indexed_dictionary(src_wv, trg_wv), dtype=xp.int32)

    if args.init_num:
        init_dict = get_numeral_init_dict(src_wv, trg_wv)
    elif args.unsupervised:
        init_dict = get_unsupervised_init_dict(src_emb, trg_emb, args.vocab_cutoff, args.csls, args.normalize, args.direction)
    else:
        init_dict = xp.array(utils.BilingualDict(args.dictionary).get_indexed_dictionary(src_wv, trg_wv), dtype=xp.int32)
    del src_wv, trg_wv

    # allocate memory for large matrices
    src_size = src_emb.shape[0]
    trg_size = trg_emb.shape[0]
    fwd_trg_size = min(trg_size, args.vocab_cutoff) if args.target_cutoff else trg_size
    bwd_src_size = min(src_size, args.vocab_cutoff) if args.source_cutoff else src_size
    fwd_sim = xp.zeros((args.batch_size, fwd_trg_size), dtype=xp.float32)
    bwd_sim = xp.zeros((args.batch_size, bwd_src_size), dtype=xp.float32)
    val_sim = xp.zeros((args.val_batch_size, trg_size), dtype=xp.float32)
    fwd_src_size = min(src_size, args.vocab_cutoff) if args.vocab_cutoff > 0 else src_size
    bwd_trg_size = min(trg_size, args.vocab_cutoff) if args.vocab_cutoff > 0 else trg_size
    fwd_ind = xp.arange(fwd_src_size, dtype=xp.int32)
    bwd_ind = xp.arange(bwd_trg_size, dtype=xp.int32)
    fwd_trg_ind = xp.arange(fwd_src_size, dtype=xp.int32)
    bwd_src_ind = xp.arange(bwd_trg_size, dtype=xp.int32)
    curr_dict = init_dict
    trg_proj_emb = xp.empty((trg_size, args.vector_dim), dtype=xp.float32)
    keep_prob = args.dropout
    
    logging.info('src_emb_size: %d, trg_emb_size: %d, init_dict_size: %d, gold_dict_size: %d, fwd_trg_size: %d, bwd_src_size: %d' % (src_size, trg_size, init_dict.shape[0], gold_dict.shape[0], fwd_trg_size, bwd_src_size))
    logging.info('=============================================================================')

    # self learning
    for epoch in range(args.epochs):
        # calculate W_trg
        X_src = src_emb[curr_dict[:, 0]]
        X_trg = trg_emb[curr_dict[:, 1]]
        if args.W_target != '' and epoch == 0:
            with open(args.W_target, 'rb') as fin:
                W_trg = pickle.load(fin)
        else:
            W_trg = get_W_target(X_src, X_trg, orthogonal=args.orthogonal)
        xp.dot(trg_emb, W_trg, out=trg_proj_emb)

        # dictionary induction
        bs = args.batch_size
        if args.direction in ('forward', 'union'):
            knn_sim_fwd = xp.zeros(fwd_trg_size, dtype=xp.float32)
            if args.csls > 0:
                for i in range(0, fwd_trg_size, bs):
                    j = min(i + bs, fwd_trg_size)
                    xp.dot(trg_proj_emb[i:j], src_emb[:fwd_src_size].T, out=bwd_sim[:j - i])
                    knn_sim_fwd[i:j] = top_k_mean(bwd_sim[:j - i], k=args.csls, inplace=True)
            for i in range(0, fwd_src_size, bs):
                j = min(fwd_src_size, i + bs)
                xp.dot(src_emb[fwd_ind[i:j]], trg_proj_emb[:fwd_trg_size].T, out=fwd_sim[:j - i])
                fwd_sim[:j - i] -= knn_sim_fwd / 2
                dropout(fwd_sim[:j - i], keep_prob).argmax(axis=1, out=fwd_trg_ind[i:j])
        if args.direction in ('backward', 'union'):
            knn_sim_bwd = xp.zeros(bwd_src_size, dtype=xp.float32)
            if args.csls > 0:
                for i in range(0, bwd_src_size, bs):
                    j = min(i + bs, bwd_src_size)
                    xp.dot(src_emb[i:j], trg_proj_emb[:bwd_trg_size].T, out=fwd_sim[:j - i])
                    knn_sim_bwd[i:j] = top_k_mean(fwd_sim[:j - i], k=args.csls, inplace=True)
            for i in range(0, bwd_trg_size, bs):
                j = min(bwd_trg_size, i + bs)
                xp.dot(trg_proj_emb[bwd_ind[i:j]], src_emb[:bwd_src_size].T, out=bwd_sim[:j - i])
                bwd_sim[:j - i] -= knn_sim_bwd / 2
                dropout(bwd_sim[:j - i], keep_prob).argmax(axis=1, out=bwd_src_ind[i:j])
        if args.direction == 'forward':
            curr_dict = xp.stack([fwd_ind, fwd_trg_ind], axis=1)
        elif args.direction == 'backward':
            curr_dict = xp.stack([bwd_src_ind, bwd_ind], axis=1)
        elif args.direction == 'union':
            curr_dict = xp.stack([xp.concatenate((fwd_ind, bwd_src_ind)), xp.concatenate((fwd_trg_ind, bwd_ind))], axis=1)
        
        if epoch % 4 == 0:
            keep_prob = min(1., keep_prob + 0.1)

        # valiadation
        bs = args.val_batch_size
        if not args.no_valiadation or epoch == (args.epochs - 1):
            val_trg_indices = xp.zeros(gold_dict.shape[0], dtype=xp.int32)
            for i in range(0, gold_dict.shape[0], bs):
                j = min(gold_dict.shape[0], i + bs)
                xp.dot(src_emb[gold_dict[i:j, 0]], trg_proj_emb.T, out=val_sim[:j - i])
                xp.argmax(val_sim[:j - i], axis=1, out=val_trg_indices[i:j])
            accuracy = xp.mean((val_trg_indices == gold_dict[:, 1]).astype(xp.int32))
            logging.info('epoch: %d   accuracy: %.4f   dict_size: %d' % (epoch, accuracy, curr_dict.shape[0]))
            log_file.write('%d,%.4f\n' % (epoch, accuracy))

    log_file.close()

    # save W_trg
    with open(args.save_path, 'wb') as fout:
        pickle.dump(W_trg, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sl', '--source_lang', default='en', help='source language: en/es/ca/eu (default: en)')
    parser.add_argument('-tl', '--target_lang', default='es', help='target language: en/es/ca/eu (default: es)')
    parser.add_argument('-se', '--source_embedding', default='./emb/en.bin', help='monolingual word embedding of the source language (default: ./emb/en.bin)')
    parser.add_argument('-te', '--target_embedding', default='./emb/es.bin', help='monolingual word embedding of the target language (default: ./emb/es.bin)')
    parser.add_argument('-gd', '--gold_dictionary', default='./lexicons/apertium/en-es.txt', help='gold bilingual dictionary for evaluation(default: ./lexicons/apertium/en-es.txt)')
    parser.add_argument('-W', '--W_target', type=str, default='', help='restore W_target from a file')
    parser.add_argument('-vd', '--vector_dim', default=300, type=int, help='dimension of each word vector (default: 300)')

    parser.add_argument('-e', '--epochs', default=50, type=int, help='training epochs (default: 50)')
    parser.add_argument('-bs', '--batch_size', default=1000, type=int, help='training batch size (default: 1000)')
    parser.add_argument('-vbs', '--val_batch_size', default=20, type=int, help='training batch size (default: 20)')

    parser.add_argument('--orthogonal', action='store_true', help='restrict projection matrix to be orthogonal')
    parser.add_argument('-vc', '--vocab_cutoff', default=20000, type=int, help='restrict the vocabulary to k most frequent words')
    parser.add_argument('-tc', '--target_cutoff', action='store_true', help='target vocab cutoff during training')
    parser.add_argument('-sc', '--source_cutoff', action='store_true', help='source vocab cutoff during training')
    parser.add_argument('--csls', type=int, default=10, help='number of csls neighbours')
    parser.add_argument('--dropout', type=float, default=0.1, help='initial keep prob of the dropout machanism')
    parser.add_argument('--normalize', choices=['unit', 'center', 'unitdim', 'centeremb', 'none'], nargs='*', default=['unit', 'center', 'unit'], help='normalization actions')
    parser.add_argument('--direction', choices=['forward', 'backward', 'union'], default='union', help='direction of dictionary induction')

    parser.add_argument('--no_valiadation', action='store_true', help='disable valiadation at each iteration')
    parser.add_argument('--debug', action='store_const', dest='loglevel', default=logging.INFO, const=logging.DEBUG, help='print debug info')
    parser.add_argument('--save_path', default='./checkpoints/wtarget.bin', help='file to save the learned W_target')
    parser.add_argument('--cuda', action='store_true', help='use cuda to accelerate')
    parser.add_argument('--log', default='./log/init100.csv', type=str, help='file to print log')
    parser.add_argument('--plot', action='store_true', help='plot results')

    parser.add_argument('--recommend', action='store_true', help='use recommended settings')

    init_dict_group = parser.add_mutually_exclusive_group()
    init_dict_group.add_argument('-d', '--dictionary', default='./init_dict/init100.txt', help='bilingual dictionary for learning bilingual mapping (default: ./init_dict/init100.txt)')
    init_dict_group.add_argument('--init_num', action='store_true', help='use numerals as initial dictionary')
    init_dict_group.add_argument('--unsupervised', action='store_true', help='use unsupervised init')

    args = parser.parse_args()
    if args.recommend:
        parser.set_defaults(unsupervised=True, target_cutoff=True, source_cutoff=True, csls=10, direction='union', cuda=True, normalize=['center', 'unit'], epochs=50, batch_size=1000, val_bach_size=20, vocab_cutoff=20000, orthogonal=True)
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
