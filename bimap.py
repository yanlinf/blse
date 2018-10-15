import argparse
import pickle
import logging
import sys
import os
import re
import numpy as np
from utils.utils import *
from cupy_utils import *


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


def get_numeral_init_dict(src_wv, trg_wv):
    num_regex = re.compile('^[0-9]+$')
    src_nums = {w for w in src_wv.vocab if num_regex.match(w) is not None}
    trg_nums = {w for w in trg_wv.vocab if num_regex.match(w) is not None}
    common = src_nums & trg_nums
    init_dict = xp.array([[src_wv.word2index(w), trg_wv.word2index(w)] for w in common], dtype=xp.int32)
    return init_dict


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

    src_wv = WordVecs(args.source_embedding, emb_format=args.format).normalize(args.normalize)
    trg_wv = WordVecs(args.target_embedding, emb_format=args.format).normalize(args.normalize)
    src_emb = xp.array(src_wv.embedding, dtype=xp.float32)
    trg_emb = xp.array(trg_wv.embedding, dtype=xp.float32)
    gold_dict = xp.array(BilingualDict(args.gold_dictionary).get_indexed_dictionary(src_wv, trg_wv), dtype=xp.int32)
    keep_prob = args.dropout_init
    
    if args.init_num:
        init_dict = get_numeral_init_dict(src_wv, trg_wv)
    elif args.init_unsupervised:
        init_dict = get_unsupervised_init_dict(src_emb, trg_emb, args.vocab_cutoff, args.csls, args.normalize, args.direction)
    else:
        init_dict = xp.array(BilingualDict(args.init_dictionary).get_indexed_dictionary(src_wv, trg_wv), dtype=xp.int32)
    curr_dict = init_dict
    del src_wv, trg_wv

    bdi_obj = BDI(src_emb, trg_emb, batch_size=args.batch_size, cutoff_size=args.vocab_cutoff, cutoff_type='both', direction=args.direction, csls=args.csls, batch_size_val=args.val_batch_size)

    # self learning
    for epoch in range(args.epochs):
        # calculate W_trg
        X_src = src_emb[curr_dict[:, 0]]
        X_trg = trg_emb[curr_dict[:, 1]]
        if args.W_target != '' and epoch == 0:
            with open(args.W_target, 'rb') as fin:
                W_trg = pickle.load(fin)
        else:
            W_trg = get_projection_matrix(X_src, X_trg, orthogonal=args.orthogonal, direction='backward')
        bdi_obj.project(W_trg)

        # dictionary induction
        curr_dict = bdi_obj.get_bilingual_dict_with_cutoff(keep_prob=keep_prob)

        # update keep_prob
        if (epoch + 1) % args.dropout_interval == 0:
            keep_prob = min(1., keep_prob + args.dropout_step)

        # valiadation
        if not args.no_valiadation and (epoch + 1) % args.valiadation_step == 0 or epoch == (args.epochs - 1):
            bdi_obj.project(W_trg, full_trg=True)
            val_trg_ind = bdi_obj.get_target_indices(gold_dict[:, 0])
            accuracy = xp.mean((val_trg_ind == gold_dict[:, 1]).astype(xp.int32))
            logging.info('epoch: %d   accuracy: %.4f   dict_size: %d' % (epoch, accuracy, curr_dict.shape[0]))
            log_file.write('%d,%.4f\n' % (epoch, accuracy))

    log_file.close()

    # save W_trg
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    with open(args.save_path, 'wb') as fout:
        pickle.dump(asnumpy(W_trg), fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sl', '--source_lang', default='en', help='source language: en/es/ca/eu (default: en)')
    parser.add_argument('-tl', '--target_lang', default='es', help='target language: en/es/ca/eu (default: es)')
    parser.add_argument('-se', '--source_embedding', default='./emb/en.bin', help='monolingual word embedding of the source language (default: ./emb/en.bin)')
    parser.add_argument('-te', '--target_embedding', default='./emb/es.bin', help='monolingual word embedding of the target language (default: ./emb/es.bin)')
    parser.add_argument('--format', choices=['word2vec_bin', 'fasttext_text'], default='word2vec_bin', help='word embedding format')
    parser.add_argument('-gd', '--gold_dictionary', default='./lexicons/apertium/en-es.txt', help='gold bilingual dictionary for evaluation(default: ./lexicons/apertium/en-es.txt)')
    parser.add_argument('-W', '--W_target', type=str, default='', help='restore W_target from a file')
    parser.add_argument('-vd', '--vector_dim', default=300, type=int, help='dimension of each word vector (default: 300)')
    parser.add_argument('-e', '--epochs', default=500, type=int, help='training epochs (default: 500)')
    parser.add_argument('-bs', '--batch_size', default=5000, type=int, help='training batch size (default: 5000)')
    parser.add_argument('-vbs', '--val_batch_size', default=1000, type=int, help='training batch size (default: 1000)')
    parser.add_argument('--no_valiadation', action='store_true', help='disable valiadation at each iteration')
    parser.add_argument('--valiadation_step', type=int, default=50, help='valiadation frequency')
    parser.add_argument('--debug', action='store_const', dest='loglevel', default=logging.INFO, const=logging.DEBUG, help='print debug info')
    parser.add_argument('--save_path', default='./checkpoints/wtarget.bin', help='file to save the learned W_target')
    parser.add_argument('--cuda', action='store_true', help='use cuda to accelerate')
    parser.add_argument('--log', default='./log/init100.csv', type=str, help='file to print log')
    parser.add_argument('--plot', action='store_true', help='plot results')

    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument('-d', '--init_dictionary', default='./init_dict/init100.txt', help='bilingual dictionary for learning bilingual mapping (default: ./init_dict/init100.txt)')
    init_group.add_argument('--init_num', action='store_true', help='use numerals as initial dictionary')
    init_group.add_argument('--init_unsupervised', action='store_true', help='use unsupervised init')

    mapping_group = parser.add_argument_group()
    mapping_group.add_argument('--orthogonal', action='store_true', help='restrict projection matrix to be orthogonal')
    mapping_group.add_argument('--normalize', choices=['unit', 'center', 'unitdim', 'centeremb', 'none'], nargs='*', default=['unit', 'center', 'unit'], help='normalization actions')

    induction_group = parser.add_argument_group()
    induction_group.add_argument('-vc', '--vocab_cutoff', default=10000, type=int, help='restrict the vocabulary to k most frequent words')
    induction_group.add_argument('--csls', type=int, default=10, help='number of csls neighbours')
    induction_group.add_argument('--dropout_init', type=float, default=0.1, help='initial keep prob of the dropout machanism')
    induction_group.add_argument('--dropout_interval', type=int, default=30, help='increase keep_prob every m steps')
    induction_group.add_argument('--dropout_step', type=float, default=0.1, help='increase keep_prob by a small step')
    induction_group.add_argument('--direction', choices=['forward', 'backward', 'union'], default='union', help='direction of dictionary induction')

    recommend_group = parser.add_mutually_exclusive_group()
    recommend_group.add_argument('-u', '--unsupervised', action='store_true', help='use unsupervised settings')
    recommend_group.add_argument('-uc', '--unconstrained', action='store_true', help='use unsupervised and unconstrained settings')
    recommend_group.add_argument('-s5', '--supervised5000', action='store_true', help='use supervised5000 settings')
    recommend_group.add_argument('-s1', '--supervised100', action='store_true', help='use supervised100 settings')

    lang_group = parser.add_mutually_exclusive_group()
    lang_group.add_argument('--en_es', action='store_true', help='train english-spanish embedding')
    lang_group.add_argument('--en_ca', action='store_true', help='train english-catalan embedding')
    lang_group.add_argument('--en_eu', action='store_true', help='train english-basque embedding')

    args = parser.parse_args()
    if args.unsupervised:
        parser.set_defaults(init_unsupervised=True, csls=10, direction='union', cuda=True, normalize=['center', 'unit'], vocab_cutoff=10000, orthogonal=True, log='./log/unsupervised.csv', dropout_interval=40)
    elif args.unconstrained:
        parser.set_defaults(init_unsupervised=True, csls=10, direction='union', cuda=True, normalize=['center', 'unit'], vocab_cutoff=10000, orthogonal=False, log='./log/unconstrained.csv')
    elif args.supervised5000:
        parser.set_defaults(init_dictionary='./init_dict/init5000.txt', csls=10, direction='union', cuda=True,
                            normalize=['center', 'unit'], vocab_cutoff=10000, orthogonal=True, log='./log/supervised5000.csv')
    elif args.supervised100:
        parser.set_defaults(init_dictionary='./init_dict/init100.txt', csls=10, direction='union', cuda=True,
                            normalize=['center', 'unit'], vocab_cutoff=10000, orthogonal=True, log='./log/supervised100.csv')

    if args.en_es:
        parser.set_defaults(source_embedding='emb/wiki.en.vec', target_embedding='emb/wiki.es.vec', format='fasttext_text',
                            source_dataset='datasets/en/opener_sents/', target_dataset='datasets/es/opener_sents/',
                            gold_dictionary='lexicons/apertium/en-es.txt', save_path='checkpoints/en-es-ubi-0.bin')
    elif args.en_ca:
        parser.set_defaults(source_embedding='emb/wiki.en.vec', target_embedding='emb/wiki.ca.vec', format='fasttext_text',
                            source_dataset='datasets/en/opener_sents/', target_dataset='datasets/ca/opener_sents/',
                            gold_dictionary='lexicons/apertium/en-ca.txt', save_path='checkpoints/en-ca-ubi-0.bin')
    elif args.en_eu:
        parser.set_defaults(source_embedding='emb/wiki.en.vec', target_embedding='emb/wiki.eu.vec', format='fasttext_text',
                            source_dataset='datasets/en/opener_sents/', target_dataset='datasets/eu/opener_sents/',
                            gold_dictionary='lexicons/apertium/en-eu.txt', save_path='checkpoints/en-eu-ubi-0.bin')
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
