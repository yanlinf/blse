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


def get_pos_neg_vecs(X, y):
    xp = get_array_module(X, y)
    pos_mask = y <= 1
    xpos = X[pos_mask]
    xneg = X[~pos_mask]
    return xpos, xneg


def get_projection_with_senti(X_src, X_trg, alpha, pos=None, neg=None, pos1=None, pos2=None, neg1=None, neg2=None,
                              direction='forward', orthogonal=False, normalize=False, spectral=False,
                              threshold=1., learning_rate=0):
    xp = get_array_module(X_src, X_trg, pos, neg)
    logging.debug('alpha: %.4f' % alpha)
    if orthogonal:
        if direction == 'forward':
            u, s, vt = xp.linalg.svd(xp.dot(X_trg.T, X_src))
            W = xp.dot(vt.T, u.T)
        elif direction == 'backward':
            u, s, vt = xp.linalg.svd(xp.dot(X_src.T, X_trg))
            W = xp.dot(vt.T, u.T)
    elif learning_rate > 0:
        if direction == 'forward':
            W = xp.linalg.pinv(X_src.T.dot(X_src) -
                               alpha * (pos - neg).T.dot(pos - neg) +
                               alpha * (pos1 - pos2).T.dot(pos1 - pos1) +
                               alpha * (neg1 - neg2).T.dot(neg1 - neg2)).dot(X_src.T.dot(X_trg))
            if spectral:
                W = proj_spectral(W)
            prev_loss = float('inf')
            for i in range(20):
                loss = -alpha * xp.linalg.norm((pos - neg).dot(W)) + \
                    xp.linalg.norm(X_src.dot(W) - X_trg) + \
                    alpha * xp.linalg.norm((pos1 - pos2).dot(W)) + \
                    alpha * xp.linalg.norm((neg1 - neg2).dot(W))
                logging.debug('loss: %.4f' % loss)
                if prev_loss - loss < 0.05:
                    break
                else:
                    prev_loss = loss
                grad = 2 * (-alpha * (pos - neg).T.dot(pos - neg) +
                            X_src.T.dot(X_src) +
                            alpha * (pos1 - pos2).T.dot(pos1 - pos2) +
                            alpha * (neg1 - neg2).T.dot(neg1 - neg2)).dot(W) - 2 * X_src.T.dot(X_trg)
                W -= learning_rate * grad
                if spectral:
                    W = proj_spectral(W)
        elif direction == 'backward':
            W = xp.linalg.pinv(X_trg.T.dot(X_trg)).dot(X_trg.T.dot(X_src))
            if spectral:
                W = proj_spectral(W)
            prev_loss = float('inf')
            for i in range(20):
                loss = xp.linalg.norm(X_trg.dot(W) - X_src)
                logging.debug('loss: %.4f' % loss)
                if prev_loss - loss < 0.05:
                    break
                else:
                    prev_loss = loss
                grad = 2 * (X_trg.T.dot(X_trg)).dot(W) - 2 * X_trg.T.dot(X_src)
                W -= learning_rate * grad
                if spectral:
                    W = proj_spectral(W)
    else:
        if direction == 'forward':
            W = xp.linalg.pinv(X_src.T.dot(X_src) - alpha * (pos - neg).T.dot(pos - neg)).dot(X_src.T.dot(X_trg))
        elif direction == 'backward':
            W = xp.linalg.pinv(X_trg.T.dot(X_trg)).dot(X_trg.T.dot(X_src))

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

    if not os.path.exists('log'):
        os.mkdir('log')
    log_file = open(args.log, 'w', encoding='utf-8')

    if args.pickle:
        with open(args.source_embedding, 'rb') as fin:
            src_wv = pickle.load(fin)
        with open(args.target_embedding, 'rb') as fin:
            trg_wv = pickle.load(fin)
    else:
        src_wv = WordVecs(args.source_embedding, emb_format=args.format).normalize(args.normalize)
        trg_wv = WordVecs(args.target_embedding, emb_format=args.format).normalize(args.normalize)
    src_ds = SentimentDataset(args.source_dataset).to_index(src_wv).to_vecs(src_wv.embedding)
    trg_ds = SentimentDataset(args.target_dataset).to_index(trg_wv).to_vecs(trg_wv.embedding)
    src_pos, src_neg = get_pos_neg_vecs(xp.array(src_ds.train[0]), xp.array(src_ds.train[1]))
    trg_pos, trg_neg = get_pos_neg_vecs(xp.array(trg_ds.train[0]), xp.array(trg_ds.train[1]))
    gold_dict = xp.array(BilingualDict(args.gold_dictionary).get_indexed_dictionary(src_wv, trg_wv), dtype=xp.int32)
    keep_prob = args.dropout_init
    alpha = min(args.alpha, args.alpha_init)

    logging.info('gold dict shape' + str(gold_dict.shape))

    if args.init_num:
        init_dict = get_numeral_init_dict(src_wv, trg_wv)
    elif args.init_unsupervised:
        init_dict = get_unsupervised_init_dict(src_wv.embedding, trg_wv.embedding, args.vocab_cutoff, args.csls, args.normalize, args.direction)
        init_dict = xp.array(init_dict)
    elif args.init_random:
        size = args.vocab_cutoff * 2 if args.direction == 'both' else args.vocab_cutoff
        init_dict = xp.stack((xp.arange(size), xp.random.permutation(size)), axis=1)
    else:
        init_dict = xp.array(BilingualDict(args.init_dictionary).get_indexed_dictionary(src_wv, trg_wv), dtype=xp.int32)

    if args.load is not None:
        W_src, W_trg, model_type = load_model(args.load)
        W_src = xp.array(W_src)
        W_trg = xp.array(W_trg)
    else:
        W_src = W_trg = xp.identity(args.vector_dim, dtype=xp.float32)

    bdi_obj = BDI(src_wv.embedding, trg_wv.embedding, batch_size=args.batch_size, cutoff_size=args.vocab_cutoff, cutoff_type='both',
                  direction=args.direction, csls=args.csls, batch_size_val=args.val_batch_size, scorer=args.scorer,
                  src_val_ind=gold_dict[:, 0], trg_val_ind=gold_dict[:, 1])
    bdi_obj.project(W_src, 'forward', unit_norm=args.normalize_projection)
    bdi_obj.project(W_trg, 'backward', unit_norm=args.spectral)
    curr_dict = init_dict if args.load is None else bdi_obj.get_bilingual_dict_with_cutoff(keep_prob=keep_prob)

    # self learning
    for epoch in range(args.epochs):
        logging.debug('running epoch %d...' % epoch)
        logging.debug('alhpa: %.4f' % alpha)

        if epoch % 2 == 0:
            X_src = bdi_obj.src_emb[curr_dict[:, 0]]
            X_trg = bdi_obj.trg_proj_emb[curr_dict[:, 1]]
            xpos, xneg = sample(src_pos, src_neg, args.senti_nsample)
            xpos1, xpos2 = sample(src_pos, src_pos, args.pos_nsample)
            xneg1, xneg2 = sample(src_neg, src_neg, args.neg_nsample)
            W_src = get_projection_with_senti(X_src, X_trg, alpha, xpos, xneg, xpos1, xpos2, xneg1, xneg2, 'forward', args.orthogonal,
                                              args.normalize_W, args.spectral, args.threshold,
                                              learning_rate=args.learning_rate)
            logging.debug('squared f-norm of W_src: %.4f' % xp.sum(W_src**2))
            bdi_obj.project(W_src, 'forward', unit_norm=args.normalize_projection, scale=args.scale)
            if args.scale:
                W_src *= bdi_obj.src_factor
        elif epoch % 2 == 1:
            X_src = bdi_obj.src_proj_emb[curr_dict[:, 0]]
            X_trg = bdi_obj.trg_emb[curr_dict[:, 1]]
            W_trg = get_projection_with_senti(X_src, X_trg, alpha=0, direction='backward', orthogonal=args.orthogonal,
                                              normalize=False, spectral=args.spectral, threshold=args.threshold,
                                              learning_rate=args.learning_rate)
            logging.debug('squared f-norm of W_trg: %.4f' % xp.sum(W_trg**2))
            bdi_obj.project(W_trg, 'backward', unit_norm=args.normalize_projection, scale=args.scale)
            if args.scale:
                W_trg *= bdi_obj.trg_factor

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
        else:
            alpha = min(args.alpha_factor * alpha, args.alpha)

        # valiadation
        if not args.no_valiadation and (epoch + 1) % args.valiadation_step == 0 or epoch == (args.epochs - 1):
            bdi_obj.project(W_trg, 'backward', unit_norm=args.normalize_projection, scale=args.scale, full_trg=True)
            val_trg_ind = bdi_obj.get_target_indices(gold_dict[:, 0])
            accuracy = xp.mean((val_trg_ind == gold_dict[:, 1]).astype(xp.int32))
            logging.info('epoch: %d   accuracy: %.4f   dict_size: %d' % (epoch, accuracy, curr_dict.shape[0]))
            log_file.write('%d,%.4f\n' % (epoch, accuracy))

    log_file.close()

    # save W_trg
    save_model(asnumpy(W_src), asnumpy(W_trg), args.source_lang,
               args.target_lang, args.model, args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

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
    dataset_group.add_argument('-td', '--target_dataset', default='./datasets/es/opener_sents/', help='target sentiment dataset')
    dataset_group.add_argument('-gd', '--gold_dictionary', default='./lexicons/apertium/en-es.txt', help='gold bilingual dictionary for evaluation(default: ./lexicons/apertium/en-es.txt)')

    io_group = parser.add_argument_group()
    io_group.add_argument('--load', help='restore W_src and W_trg from a file')
    io_group.add_argument('--pickle', action='store_true', help='load from pickled objects')
    io_group.add_argument('--save_path', default='./checkpoints/senti.bin', help='file to save W_src and W_trg')
    io_group.add_argument('--log', default='./log/init100.csv', type=str, help='file to print log')

    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument('-d', '--init_dictionary', default='./init_dict/init100.txt', help='bilingual dictionary for learning bilingual mapping (default: ./init_dict/init100.txt)')
    init_group.add_argument('--init_num', action='store_true', help='use numerals as initial dictionary')
    init_group.add_argument('--init_random', action='store_true', help='use random initial dictionary')
    init_group.add_argument('--init_unsupervised', action='store_true', help='use unsupervised init')

    mapping_group = parser.add_argument_group()
    mapping_group.add_argument('--normalize', choices=['unit', 'center', 'unitdim', 'centeremb', 'none'], nargs='*', default=['center', 'unit'], help='normalization actions')
    mapping_group.add_argument('--orthogonal', action='store_true', help='restrict projection matrix to be orthogonal')
    mapping_group.add_argument('--spectral', action='store_true', help='restrict projection matrix to spectral domain')
    mapping_group.add_argument('--normalize_projection', action='store_true', help='normalize after projection')
    mapping_group.add_argument('-lr', '--learning_rate', type=float, default=0, help='use gradient descent to solve W')
    mapping_group.add_argument('--scale', action='store_true', help='scale embedding after projecting')
    mapping_group.add_argument('--threshold', type=float, default=1., help='thresholding the singular value of W')
    mapping_group.add_argument('--normalize_W', action='store_true', help='add f-norm restriction on W')
    mapping_group.add_argument('--no_target_senti', action='store_true', help='no target sentiment')

    senti_sample_group = parser.add_argument_group()
    senti_sample_group.add_argument('-n', '--senti_nsample', type=int, default=200, help='sentiment examples')
    senti_sample_group.add_argument('--pos_nsample', type=int, default=100, help='positive examples')
    senti_sample_group.add_argument('--neg_nsample', type=int, default=100, help='negative examples')

    alpha_group = parser.add_argument_group()
    alpha_group.add_argument('-a', '--alpha', type=float, default=5, help='trade-off between sentiment and alignment')
    alpha_group.add_argument('--alpha_init', type=float, default=0.1, help='initial value of alpha')
    alpha_group.add_argument('--alpha_factor', type=float, default=1.01, help='multiply alpha by a factor each epoch')
    alpha_group.add_argument('--alpha_step', type=float, default=0.02, help='multiply alpha by a factor each epoch')
    alpha_update = alpha_group.add_mutually_exclusive_group()
    alpha_update.add_argument('--alpha_inc', action='store_true', help='increase alpha by a step each epoch')
    alpha_update.add_argument('--alpha_mul', action='store_true', help='multiply alpha by a factor each epoch')

    induction_group = parser.add_argument_group()
    induction_group.add_argument('-vc', '--vocab_cutoff', default=10000, type=int, help='restrict the vocabulary to k most frequent words')
    induction_group.add_argument('--csls', type=int, default=10, help='number of csls neighbours')
    induction_group.add_argument('--dropout_init', type=float, default=0.1, help='initial keep prob of the dropout machanism')
    induction_group.add_argument('--dropout_interval', type=int, default=50, help='increase keep_prob every m steps')
    induction_group.add_argument('--dropout_step', type=float, default=0.1, help='increase keep_prob by a small step')
    induction_group.add_argument('--direction', choices=['forward', 'backward', 'union'], default='union', help='direction of dictionary induction')
    induction_group.add_argument('--scorer', choices=['dot', 'cos', 'euclidean'], default='dot', help='scorer for nearest neighbour retrieval')

    recommend_group = parser.add_mutually_exclusive_group()
    recommend_group.add_argument('-u', '--unsupervised', action='store_true', help='use recommended settings')
    recommend_group.add_argument('-s5', '--supervised5000', action='store_true', help='use supervised5000 settings')
    recommend_group.add_argument('-s1', '--supervised100', action='store_true', help='use supervised100 settings')
    recommend_group.add_argument('--senti', action='store_true', help='use unsupervised + senti settings')
    recommend_group.add_argument('--ubise', action='store_true', help='use unsupervised + senti settings')
    recommend_group.add_argument('--test', action='store_true', help='use test settings')
    recommend_group.add_argument('--unconstrained', action='store_true', help='use unsupervised + unconstrained settings')

    lang_group = parser.add_mutually_exclusive_group()
    lang_group.add_argument('--en_es', action='store_true', help='train english-spanish embedding')
    lang_group.add_argument('--en_ca', action='store_true', help='train english-catalan embedding')
    lang_group.add_argument('--en_eu', action='store_true', help='train english-basque embedding')

    args = parser.parse_args()
    if args.unsupervised:
        parser.set_defaults(init_unsupervised=True, csls=10, direction='union', cuda=True, normalize=['center', 'unit'],
                            vocab_cutoff=10000, orthogonal=True, log='./log/unsupervised.csv', model='ubi')
    elif args.supervised5000:
        parser.set_defaults(init_dictionary='./init_dict/init5000.txt', csls=10, direction='union', cuda=True,
                            normalize=['center', 'unit'], vocab_cutoff=10000, orthogonal=True, log='./log/supervised5000.csv')
    elif args.supervised100:
        parser.set_defaults(init_dictionary='./init_dict/init100.txt', csls=10, direction='union', cuda=True,
                            normalize=['center', 'unit'], vocab_cutoff=10000, orthogonal=True, log='./log/supervised100.csv')
    elif args.senti:
        parser.set_defaults(init_unsupervised=True, csls=10, direction='union', cuda=True, normalize=['center', 'unit'],
                            vocab_cutoff=10000, alpha=7, senti_nsample=200, log='./log/senti.csv', spectral=False, threshold=1.,
                            learning_rate=0.001, alpha_init=0.1, alpha_factor=1.01, no_proj_error=False,
                            dropout_init=0.1, dropout_interval=1, dropout_step=0.002, epochs=1000, model='ubise',
                            normalize_projection=True)
    elif args.ubise:
        parser.set_defaults(init_unsupervised=True, csls=10, direction='union', cuda=True, normalize=['center', 'unit'],
                            vocab_cutoff=10000, alpha=5, senti_nsample=200, log='./log/senti.csv', spectral=True, threshold=1.,
                            learning_rate=0.001, alpha_init=0.1, alpha_step=0.02, alpha_inc=True,
                            no_proj_error=False,
                            dropout_init=0.1, dropout_interval=1, dropout_step=0.002, epochs=1000,
                            no_target_senti=True, model='ubise', normalize_projection=True,)
    elif args.unconstrained:
        parser.set_defaults(init_unsupervised=True, csls=10, direction='union', cuda=True, normalize=['center', 'unit'],
                            vocab_cutoff=10000, alpha=0.1, senti_nsample=200, log='./log/senti.csv', scorer='euclidean', scale=True)

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
