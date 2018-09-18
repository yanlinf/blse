import numpy as np
import argparse
import pickle
import logging
import sys
import re
from utils import utils

try:
    import cupy
except ImportError:
    cupy = None

def main(args):
    logging.info(str(args))

    if args.cuda:
        if cupy is None:
            print('Install cupy for cuda support')
            sys.exit(-1)
        xp = cupy
    else:
        xp = np
    
    # prepare initial word vectors
    src_wv = utils.WordVecs(args.source_embedding).mean_center().normalize()
    trg_wv = utils.WordVecs(args.target_embedding).mean_center().normalize()

    # prepare gold bilingual dict
    gold_dict = xp.array(utils.BilingualDict(args.gold_dictionary).get_indexed_dictionary(src_wv, trg_wv), dtype=xp.int32)
    
    # prepare initial bilingual dict
    if args.init_num:
        num_regex = re.compile('^[0-9]+$')
        src_nums = {w for w in src_wv.vocab if num_regex.match(w) is not None}
        trg_nums = {w for w in trg_wv.vocab if num_regex.match(w) is not None}
        print(src_nums)
        print(trg_nums)
        common = src_nums & trg_nums
        init_dict = xp.array([[src_wv.word2index(w), trg_wv.word2index(w)] for w in common], dtype=xp.int32)
    else:
        init_dict = xp.array(utils.BilingualDict(args.dictionary).get_indexed_dictionary(src_wv, trg_wv), dtype=xp.int32)
    print('init_dict_size: %d' % init_dict.shape[0])

    vocab_size = src_wv.embedding.shape[0]
    print('vocab_size: %d' % vocab_size)
    curr_dict = init_dict
    cos_sims = xp.empty((args.batch_size, trg_wv.embedding.shape[0]), dtype=xp.float32)
    src_emb = xp.array(src_wv.embedding, dtype=xp.float32)
    trg_original_emb = xp.array(trg_wv.embedding, dtype=xp.float32)
    trg_emb = xp.empty((trg_wv.embedding.shape[0], args.vector_dim), dtype=xp.float32)

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

        
        curr_dict = xp.zeros((vocab_size, 2), dtype=xp.int32)
        curr_dict[:, 0] = xp.arange(vocab_size)
        for i in range(0, vocab_size, args.batch_size):
            print(i)
            j = i + args.batch_size
            xp.dot(src_emb[i:j], trg_emb.T, out=cos_sims)  # shape (BATCH_SIZE, TARGET_VOCAB_SIZE)
            xp.argmax(cos_sims, axis=1, out=curr_dict[i:j, 1])

        accuracy = xp.mean((curr_dict[gold_dict[:, 0]][:, 1] == gold_dict[:, 1]).astype(xp.int32))
        logging.info('epoch: %d   accuracy: %.4f   dict_size: %d' % (epoch, accuracy, curr_dict.shape[0]))

    with open(args.save_path, 'wb') as fout:
        pickle.dump(W_trg, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sl', '--source_lang',
                        help='source language: en/es/ca/eu (default: en)',
                        default='en')
    parser.add_argument('-tl', '--target_lang',
                        help='target language: en/es/ca/eu (default: es)',
                        default='es')
    parser.add_argument('-se', '--source_embedding',
                        help='monolingual word embedding of the source language (default: ./emb/en.bin)',
                        default='./emb/en.bin')
    parser.add_argument('-te', '--target_embedding',
                        help='monolingual word embedding of the target language (default: ./emb/es.bin)',
                        default='./emb/es.bin')
    parser.add_argument('-gd', '--gold_dictionary',
                        help='gold bilingual dictionary for evaluation(default: ./lexicons/apertium/en-es.txt)',
                        default='./lexicons/apertium/en-es.txt')
    parser.add_argument('-e', '--epochs',
                        help='training epochs (default: 50)',
                        default=50,
                        type=int)
    parser.add_argument('-vd', '--vector_dim',
                        help='dimension of each word vector (default: 300)',
                        default=300,
                        type=int)
    parser.add_argument('-W', '--W_target',
                        help='restore W_target from a file',
                        type=str,
                        default='')
    parser.add_argument('-bs', '--batch_size',
                        help='training batch size (default: 5000)',
                        default=5000,
                        type=int)
    parser.add_argument('--debug',
                        help='print debug info',
                        action='store_const',
                        dest='loglevel',
                        default=logging.INFO,
                        const=logging.DEBUG)
    parser.add_argument('--save_path',
                        help='file to save the learned W_target',
                        default='./checkpoints/boostrap.txt')
    parser.add_argument('--cuda',
                        help='use cuda to accelerate',
                        action='store_true')

    dict_group = parser.add_mutually_exclusive_group()
    dict_group.add_argument('-d', '--dictionary',
                            help='bilingual dictionary for learning bilingual mapping (default: ./init_dict/init100.txt)',
                            default='./init_dict/init100.txt')
    dict_group.add_argument('--init_num', 
                            action='store_true',
                            help='use numerals as initial dictionary')

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel,
                        format='%(asctime)s: %(levelname)s: %(message)s')

    main(args)
