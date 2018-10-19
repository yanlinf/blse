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

    if args.pickle:
        with open(args.source_embedding, 'rb') as fin:
            src_wv = pickle.load(fin)
        with open(args.target_embedding, 'rb') as fin:
            trg_wv = pickle.load(fin)
    else:
        src_wv = WordVecs(args.source_embedding, emb_format=args.format).normalize(args.normalize)
        trg_wv = WordVecs(args.target_embedding, emb_format=args.format).normalize(args.normalize)

    pad_id = src_wv.add_word('<pad>', np.zeros(args.vector_dim, dtype=np.float32))
    src_ds = SentimentDataset(args.source_dataset).to_index(src_wv, binary=True).pad(pad_id)
    xsenti = xp.array(src_ds.train[0])
    ysenti = xp.array(src_ds.train[1])
    lsenti = xp.array(src_ds.train[2])

    gold_dict = xp.array(BilingualDict(args.gold_dictionary).get_indexed_dictionary(src_wv, trg_wv), dtype=xp.int32)
    keep_prob = args.dropout_init
    alpha = max(args.alpha, args.alpha_init) if args.alpha_dec else min(args.alpha, args.alpha_init)
    init_dict = get_unsupervised_init_dict(src_wv.embedding, trg_wv.embedding, args.vocab_cutoff, args.csls, args.normalize, args.direction)
    init_dict = xp.array(init_dict)

    logging.info('gold dict shape' + str(gold_dict.shape))

    if args.load is not None:
        dic = load_model(args.load)
        W_src = xp.array(dic['W_source'])
        W_trg = xp.array(dic['W_target'])
    else:
        W_src = W_trg = xp.identity(args.vector_dim, dtype=xp.float32)

    src_val_ind = xp.array(np.union1d(asnumpy(gold_dict[:, 0]), asnumpy(xsenti)))
    bdi_obj = BDI(src_wv.embedding, trg_wv.embedding, batch_size=args.batch_size, cutoff_size=args.vocab_cutoff, cutoff_type='both',
                  direction=args.direction, csls=args.csls, batch_size_val=args.val_batch_size, scorer='dot',
                  src_val_ind=src_val_ind, trg_val_ind=gold_dict[:, 1])
    bdi_obj.project(W_src, 'forward', unit_norm=True)
    bdi_obj.project(W_trg, 'backward', unit_norm=True)
    curr_dict = init_dict if args.load is None else bdi_obj.get_bilingual_dict_with_cutoff(keep_prob=keep_prob)

    # self learning
    try:
        for epoch in range(args.epochs):
            logging.debug('running epoch %d...' % epoch)
            logging.debug('alhpa: %.4f' % alpha)

            if epoch % 2 == 0:
                if args.loss == 0:
                    m = args.senti_nsample
                    lr = args.learning_rate
                    X_src = bdi_obj.src_emb[curr_dict[:, 0]]
                    X_trg = bdi_obj.trg_proj_emb[curr_dict[:, 1]]
                    ssrc = xp.random.randint(0, xsenti.shape[0], m)
                    strg = xp.random.randint(0, xsenti.shape[0], m)
                    if args.sample_type == 'full':
                        I = (ysenti[ssrc] == ysenti[strg]).astype(xp.float32) * 2 - 1
                    elif args.sample_type == 'same':
                        I = (ysenti[ssrc] == ysenti[strg]).astype(xp.float32)
                    elif args.sample_type == 'pos-neg':
                        I = (ysenti[ssrc] == ysenti[strg]).astype(xp.float32) - 1

                    U_src = bdi_obj.src_emb[xsenti[ssrc]].sum(axis=1) / lsenti[ssrc][:, xp.newaxis]
                    U_trg = bdi_obj.src_proj_emb[xsenti[strg]].sum(axis=1) / lsenti[strg][:, xp.newaxis]
                    U_src *= I[:, xp.newaxis]
                    logging.debug('number of samples: {0:d}'.format(U_src.shape[0]))
                    prev_loss, loss = float('inf'), float('inf')
                    while prev_loss - loss > 0.05 or loss == float('inf'):
                        prev_W = W_src.copy()
                        grad = -2 * X_src.T.dot(X_trg) - (alpha / m) * U_src.T.dot(U_trg)
                        W_src -= lr * grad
                        W_src = proj_spectral(W_src)
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
                    while prev_loss - loss > 0.05 or loss == float('inf'):
                        prev_W = W_src.copy()
                        prev_loss = loss
                        grad = 2 * ((X_src.T.dot(X_src) + (alpha / m) * Z.T.dot(Z)).dot(W_src) - X_src.T.dot(X_trg))
                        W_src -= lr * grad
                        W_src = proj_spectral(W_src)
                        loss = xp.linalg.norm(X_src.dot(W_src) - X_trg) + (alpha / m) * xp.linalg.norm(Z.dot(W_src))
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
                    while prev_loss - loss > 0.05 or loss == float('inf'):
                        prev_W = W_src.copy()
                        prev_loss = loss
                        grad = -2 * X_src.T.dot(X_trg) + (2 * alpha / m) * Z.T.dot(Z).dot(W_src)
                        W_src -= lr * grad
                        W_src = proj_spectral(W_src)
                        loss = -2 * (X_src.dot(W_src) * X_trg).sum() + (alpha / m) * xp.linalg.norm(Z.dot(W_src))
                        logging.debug('loss: {0:.4f}'.format(float(loss)))
                    if loss > prev_loss:
                        W_src = prev_W

                logging.debug('squared f-norm of W_src: %.4f' % xp.sum(W_src**2))
                bdi_obj.project(W_src, 'forward', unit_norm=True)

            elif epoch % 2 == 1:
                lr = args.learning_rate
                X_src = bdi_obj.src_proj_emb[curr_dict[:, 0]]
                X_trg = bdi_obj.trg_emb[curr_dict[:, 1]]
                prev_loss, loss = float('inf'), float('inf')
                while prev_loss - loss > 0.05 or loss == float('inf'):
                    prev_W = W_trg.copy()
                    grad = -X_trg.T.dot(X_src) if args.loss in (0, 2) else 2 * (X_trg.T.dot(X_trg).dot(W_trg) - X_trg.T.dot(X_src))
                    W_trg -= lr * grad
                    W_trg = proj_spectral(W_trg)
                    prev_loss = loss
                    loss = -(X_trg.dot(W_trg) * X_src).sum() if args.loss in (0, 2) else xp.linalg.norm(X_trg.dot(W_trg) - X_src)
                    logging.debug('loss: {0:.4f}'.format(float(loss)))
                if loss > prev_loss:
                    W_trg = prev_W
                logging.debug('squared f-norm of W_trg: %.4f' % xp.sum(W_trg**2))
                bdi_obj.project(W_trg, 'backward', unit_norm=True)

            if not args.no_proj_error:
                proj_error = xp.sum((bdi_obj.src_proj_emb[gold_dict[:, 0]] - bdi_obj.trg_proj_emb[gold_dict[:, 1]])**2)
                logging.info('proj error: %.4f' % proj_error)

            # dictionary induction
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

            # valiadation
            if not args.no_valiadation and (epoch + 1) % args.valiadation_step == 0 or epoch == (args.epochs - 1):
                bdi_obj.project(W_trg, 'backward', unit_norm=True, full_trg=True)
                val_trg_ind = bdi_obj.get_target_indices(gold_dict[:, 0])
                accuracy = xp.mean((val_trg_ind == gold_dict[:, 1]).astype(xp.int32))
                logging.info('epoch: %d   accuracy: %.4f   dict_size: %d' % (epoch, accuracy, curr_dict.shape[0]))
    finally:
        # save W_trg
        save_model(asnumpy(W_src), asnumpy(W_trg), args.source_lang,
                   args.target_lang, args.model, args.save_path,
                   alpha=args.alpha, alpha_init=args.alpha_init, dropout_init=args.dropout_init)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=int, choices=[0, 1, 2], default=0, help='type of loss function')

    training_group = parser.add_argument_group()
    training_group.add_argument('--source_lang', default='en', help='source language')
    training_group.add_argument('--target_lang', default='es', help='target language')
    training_group.add_argument('--model', choices=['ubi', 'ubise'], help='model type')
    training_group.add_argument('-e', '--epochs', default=500, type=int, help='training epochs (default: 500)')
    training_group.add_argument('-bs', '--batch_size', default=3000, type=int, help='training batch size (default: 3000)')
    training_group.add_argument('-vbs', '--val_batch_size', default=500, type=int, help='training batch size (default: 300)')
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
    parser.set_defaults(init_unsupervised=True, csls=10, direction='union', cuda=True, normalize=['center', 'unit'],
                        vocab_cutoff=10000, alpha=5000, senti_nsample=50, spectral=True, threshold=1.,
                        learning_rate=0.001, alpha_init=5000, alpha_step=0.01, alpha_inc=True,
                        no_proj_error=False, save_path='checkpoints/cvxse.bin',
                        dropout_init=1, dropout_interval=1, dropout_step=0.002, epochs=1000,
                        no_target_senti=True, model='ubise', normalize_projection=True,)

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
