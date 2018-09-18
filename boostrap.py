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

    source_wordvecs = utils.WordVecs(args.source_embedding).mean_center().normalize()
    target_wordvecs = utils.WordVecs(args.target_embedding).mean_center().normalize()
    gold_dict = utils.BilingualDict(args.gold_dictionary).get_indexed_dictionary(source_wordvecs, target_wordvecs)
    if args.init_num:
        num_regex = re.compile('$[0-9]+^')
        src_nums = {w for w in source_wordvecs.vocab if num_regex.match(w) is not None}
        trg_nums = {w for w in target_wordvecs.vocab if num_regex.match(w) is not None}
        common = src_nums & trg_nums
        init_dict = xp.array([[source_wordvecs.word2index(w), target_wordvecs.word2index(w)] for w in common], dtype=xp.int32)
    else:
        init_dict = utils.BilingualDict(args.dictionary).get_indexed_dictionary(source_wordvecs, target_wordvecs)
    print('init_dict_size: %d' % init_dict.shape[0])

    vocab_size = source_wordvecs.embedding.shape[0]
    print('vocab_size: %d' % vocab_size)
    curr_dict = init_dict
    cos_sims = xp.empty((args.batch_size, target_emb.shape[0]))
    source_emb = source_wordvecs.embedding
    target_emb = xp.empty((target_emb.shape[0], args.vector_dim))

    for epoch in range(args.epochs):
        X_source = source_wordvecs.embedding[curr_dict[:, 0]]
        X_target = target_wordvecs.embedding[curr_dict[:, 1]]
        
        if args.W_target != '' and epoch == 0:
            with open(args.W_target, 'rb') as fin:
                W_target = pickle.load(fin)
        else:
            u, s, vt = xp.linalg.svd(xp.dot(X_source.T, X_target))
            W_target = xp.dot(vt.T, u.T)
        xp.dot(target_wordvecs.embedding, W_target, out=target_emb)

        
        curr_dict = xp.zeros((vocab_size, 2), dtype=xp.int32)
        curr_dict[:, 0] = xp.arange(vocab_size)
        for i in range(0, vocab_size, args.batch_size):
            print(i)
            j = i + args.batch_size
            xp.dot(source_emb[i:j], target_emb.T, out=cos_sims)  # shape (BATCH_SIZE, TARGET_VOCAB_SIZE)
            xp.argmax(cos_sims, axis=1, out=curr_dict[i:j, 1])

        accuracy = xp.mean((curr_dict[gold_dict[:, 0]][:, 1] == gold_dict[:, 1]).astype(xp.int32))
        logging.info('epoch: %d   accuracy: %.4f   dict_size: %d' % (epoch, accuracy, curr_dict.shape[0]))

    with open(args.save_path, 'wb') as fout:
        pickle.dump(W_target, fout)


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
                        help='training batch size (default: 10000)',
                        default=10000,
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

    dict_group = parser.add_mutually_exclusive_group(required=True)
    dict_group.add_argument('-d', '--dictionary',
                            help='bilingual dictionary for learning bilingual mapping (default: ./lexicons/bingliu/en-es.txt)',
                            default='./lexicons/bingliu/en-es.txt')
    dict_group.add_argument('--init_num', 
                            action='store_true',
                            help='use numerals as initial dictionary')

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel,
                        format='%(asctime)s: %(levelname)s: %(message)s')

    main(args)
