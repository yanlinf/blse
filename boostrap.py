import argparse
import pickle
import logging
import sys
import re
from utils import utils


def main(args):
    logging.info(str(args))

    if args.cuda:
        try:
            import cupy
        except ImportError:
            print('Install cupy for cuda support')
            sys.exit(-1)
        xp = cupy
    else:
        import numpy
        xp = numpy

    # prepare initial word vectors
    src_wv = utils.WordVecs(args.source_embedding).mean_center().normalize()
    trg_wv = utils.WordVecs(args.target_embedding).mean_center().normalize()

    # prepare gold bilingual dict
    gold_dict = xp.array(utils.BilingualDict(args.gold_dictionary).get_indexed_dictionary(src_wv, trg_wv), dtype=xp.int32)

    log_file = open(args.log, 'w', encoding='utf-8')

    # prepare initial bilingual dict
    if args.init_num:
        num_regex = re.compile('^[0-9]+$')
        src_nums = {w for w in src_wv.vocab if num_regex.match(w) is not None}
        trg_nums = {w for w in trg_wv.vocab if num_regex.match(w) is not None}
        common = src_nums & trg_nums
        init_dict = xp.array([[src_wv.word2index(w), trg_wv.word2index(w)] for w in common], dtype=xp.int32)
    else:
        init_dict = xp.array(utils.BilingualDict(args.dictionary).get_indexed_dictionary(src_wv, trg_wv), dtype=xp.int32)

    # allocate memory for large matrices
    dict_size = src_wv.embedding.shape[0] if args.vocabulary_cutoff <= 0 else min(src_wv.embedding.shape[0], args.vocabulary_cutoff)
    trg_size = trg_wv.embedding.shape[0]
    curr_dict = init_dict
    sims = xp.empty((args.batch_size, trg_size), dtype=xp.float32)
    src_emb = xp.array(src_wv.embedding, dtype=xp.float32)
    trg_original_emb = xp.array(trg_wv.embedding, dtype=xp.float32)
    trg_emb = xp.empty((trg_wv.embedding.shape[0], args.vector_dim), dtype=xp.float32)
    print('src_emb_size: %d' % src_wv.embedding.shape[0])
    print('trg_emb_size: %d' % trg_wv.embedding.shape[0])
    print('init_dict_size: %d' % init_dict.shape[0])
    print('max_dict_size: %d' % dict_size)
    print('gold_dict_size: %d' % gold_dict.shape[0])
    print('------------------------------------------------')

    # self learning
    for epoch in range(args.epochs):
        X_src = src_emb[curr_dict[:, 0]]
        X_trg = trg_original_emb[curr_dict[:, 1]]

        if args.W_target != '' and epoch == 0:
            with open(args.W_target, 'rb') as fin:
                W_trg = pickle.load(fin)
        else:
            u, s, vt = xp.linalg.svd(xp.dot(X_src.T, X_trg))
            W_trg = xp.dot(vt.T, u.T)
        xp.dot(trg_original_emb, W_trg, out=trg_emb)

        curr_dict = xp.zeros((dict_size, 2), dtype=xp.int32)
        curr_dict[:, 0] = xp.arange(dict_size)
        for i in range(0, dict_size, args.batch_size):
            print('processed %d entries' % i)
            j = min(dict_size, i + args.batch_size)
            xp.dot(src_emb[i:j], trg_emb.T, out=sims[:j-i])  # shape (BATCH_SIZE, TARGET_VOCAB_SIZE)
            xp.argmax(sims[:j-i], axis=1, out=curr_dict[i:j, 1])

        # valiadation
        if not args.no_valiadation or epoch == (args.epochs - 1):
            val_trg_indices = xp.zeros(gold_dict.shape[0], dtype=xp.int32)
            for i in range(0, gold_dict.shape[0], args.batch_size):
                j = min(gold_dict.shape[0], i + args.batch_size)
                xp.dot(src_emb[gold_dict[i:j,0]], trg_emb.T, out=sims[:j-i])
                xp.argmax(sims[:j-i], axis=1, out=val_trg_indices[i:j])
            accuracy = xp.mean((val_trg_indices == gold_dict[:,1]).astype(xp.int32))
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
    parser.add_argument('-e', '--epochs', default=20, type=int, help='training epochs (default: 20)')
    parser.add_argument('-vd', '--vector_dim', default=300, type=int, help='dimension of each word vector (default: 300)')
    parser.add_argument('-W', '--W_target', type=str, default='', help='restore W_target from a file')
    parser.add_argument('-bs', '--batch_size', default=1000, type=int, help='training batch size (default: 1000)')
    parser.add_argument('-vc', '--vocabulary_cutoff', default=20000, type=int, help='restrict the vocabulary to the top k entries')
    parser.add_argument('--no_valiadation', action='store_true', help='disable valiadation at each iteration')
    parser.add_argument('--debug', action='store_const', dest='loglevel', default=logging.INFO, const=logging.DEBUG, help='print debug info')
    parser.add_argument('--save_path', default='./checkpoints/wtarget.bin', help='file to save the learned W_target')
    parser.add_argument('--cuda', action='store_true', help='use cuda to accelerate')
    parser.add_argument('--log', default='./log/init100.csv', type=str, help='file to print log')

    init_dict_group = parser.add_mutually_exclusive_group()
    init_dict_group.add_argument('-d', '--dictionary', default='./init_dict/init100.txt', help='bilingual dictionary for learning bilingual mapping (default: ./init_dict/init100.txt)')
    init_dict_group.add_argument('--init_num', action='store_true', help='use numerals as initial dictionary')

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel, format='%(asctime)s: %(levelname)s: %(message)s')

    main(args)
