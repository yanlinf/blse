import numpy as np
import argparse
import pickle
import logging
from utils import utils


BATCH_SIZE = 50

def main(args):
    logging.info(str(args))

    source_wordvecs = utils.WordVecs(args.source_embedding).mean_center().normalize()
    target_wordvecs = utils.WordVecs(args.target_embedding).mean_center().normalize()
    init_dict = utils.BilingualDict(args.dictionary).get_indexed_dictionary(
        source_wordvecs, target_wordvecs)
    gold_dict = utils.BilingualDict(args.gold_dictionary).get_indexed_dictionary(
        source_wordvecs, target_wordvecs)

    vocab_size = source_wordvecs.embedding.shape[0]
    curr_dict = init_dict
    for epoch in range(args.epochs):
        X_source = source_wordvecs.embedding[curr_dict[:, 0]]
        X_target = target_wordvecs.embedding[curr_dict[:, 1]]
        
        if args.W_target != '' and epoch == 0:
            with open(args.W_target, 'rb') as fin:
                W_target = pickle.load(fin)
        else:
            u, s, vt = np.linalg.svd(np.dot(X_source.T, X_target))
            W_target = np.dot(vt.T, u.T)

        source_emb = source_wordvecs.embedding
        target_emb = np.dot(target_wordvecs.embedding, W_target)

        
        curr_dict = np.zeros((vocab_size, 2))
        curr_dict[: 0] = np.arange(vocab_size)
        for i in range(0, vocab_size, BATCH_SIZE):
            similarities = np.dot(source_emb[i:i + BATCH_SIZE], target_emb)  # shape (BATCH_SIZE, TARGET_VOCAB_SIZE)
            closest = np.argmax(similarities, axis=1)
            curr_dict[i:i + BATCH_SIZE, 1] = closest

        accuracy = np.mean((curr_dict[gold_dict[:, 0]][:, 1] == gold_dict[:, 1]).astype(np.int32))
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
    parser.add_argument('-d', '--dictionary',
                        help='bilingual dictionary for learning bilingual mapping (default: ./lexicons/bingliu/en-es.txt)',
                        default='./lexicons/bingliu/en-es.txt')
    parser.add_argument('-gd', '--gold_dictionary',
                        help='gold bilingual dictionary for evaluation(default: ./lexicons/bingliu/en-es.txt)',
                        default='./lexicons/bingliu/en-es.txt')
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
    parser.add_argument('--debug',
                        help='print debug info',
                        action='store_const',
                        dest='loglevel',
                        default=logging.INFO,
                        const=logging.DEBUG)
    parser.add_argument('--save_path',
                        help='file to save the learned W_target',
                        default='./checkpoints/boostrap.txt')
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel,
                        format='%(asctime)s: %(levelname)s: %(message)s')

    main(args)
