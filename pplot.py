import tensorflow as tf
import numpy as np
import argparse
import pickle
from sklearn.manifold import TSNE
from utils.utils import *


def load_W_source(model_path):
    with tf.variable_scope('projection', reuse=tf.AUTO_REUSE):
        W_source = tf.get_variable(
            'W_source', dtype=tf.float32, initializer=tf.constant(np.zeros((300, 300), dtype=np.float32)))

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, model_path)
        W_source_ = sess.run(W_source)

    return W_source_


def main(args):
    with open(args.pickled_embedding, 'rb') as fin:
        wv = pickle.load(fin)
    senti_words = SentiWordSet(args.senti_words).to_index(wv)
    word_idx = sum(senti_words.wordsets, [])

    if args.model == 'ubise_source':
        if args.W != '':
            with open(args.W, 'rb') as fin:
                W, _ = pickle.load(fin)
            proj_emb = np.dot(wv.embedding, W)
            length_normalize(proj_emb, inplace=True)
        else:
            proj_emb = wv.embedding
    elif args.model == 'ubise_target':
        if args.W != '':
            with open(args.W, 'rb') as fin:
                _, W = pickle.load(fin)
            proj_emb = np.dot(wv.embedding, W)
            length_normalize(proj_emb, inplace=True)
        else:
            proj_emb = wv.embedding
    elif args.model == 'blse':
        W = load_W_source(args.W)
        proj_emb = np.dot(wv.embedding, W)

    X = proj_emb[word_idx]
    with open(args.output, 'wb') as fout:
        pickle.dump((senti_words, X), fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('W',
                        help='W_src and W_trg')
    parser.add_argument('output',
                        help='output file')
    parser.add_argument('-pe', '--pickled_embedding',
                        default='./pickle/en.bin',
                        help='pickled WordVecs object')
    parser.add_argument('-sw', '--senti_words',
                        type=str,
                        default='./categories/categories.en',
                        help='sentiment words')
    parser.add_argument('--model',
                        choices=['ubise_target', 'ubise_source', 'blse'],
                        default='ubise_target',
                        help='bilingual model')

    lang_group = parser.add_mutually_exclusive_group()
    lang_group.add_argument('--en',
                            action='store_true',
                            help='pre-plot en')
    lang_group.add_argument('--es',
                            action='store_true',
                            help='pre-plot es')
    lang_group.add_argument('--ca',
                            action='store_true',
                            help='pre-plot ca')
    lang_group.add_argument('--eu',
                            action='store_true',
                            help='pre-plot eu')
    args = parser.parse_args()

    if args.en:
        parser.set_defaults(pickled_embedding='pickle/en.bin', senti_words='categories/categories.en')
    elif args.es:
        parser.set_defaults(pickled_embedding='pickle/es.bin', senti_words='categories/categories.es')
    elif args.ca:
        parser.set_defaults(pickled_embedding='pickle/ca.bin', senti_words='categories/categories.ca')
    elif args.eu:
        parser.set_defaults(pickled_embedding='pickle/eu.bin', senti_words='categories/categories.eu')

    args = parser.parse_args()
    main(args)
