import numpy as np
import argparse
from utils.dataset import *
from utils.math import *
from utils.bdi import *
from utils.model import *
import pickle


def get_W_target(source_emb, target_emb, dict_obj, orthogonal):
    """
    source_emb: np.ndarray of shape (source_vocab_size, vec_dim)
    target_emb: np.ndarray of shape (target_vocab_size, vec_dim)
    dict_obj: np.ndarray of shape (dict_size, 2)
    orthogonal: bool

    Returns: np.ndarray of shape (vec_dim, vec_dim)
    """
    X_source = source_emb[dict_obj[:, 0]]  # shape (dict_size, vec_dim)
    X_target = target_emb[dict_obj[:, 1]]  # shape (dict_size, vec_dim)

    if orthogonal:
        u, s, vt = np.linalg.svd(np.dot(X_source.T, X_target))
        W_target = np.dot(vt.T, u.T)
    else:
        W_target = np.matmul(np.linalg.pinv(X_target), X_source)

    return W_target


def get_W_source(source_emb, target_emb, dict_obj, orthogonal):
    """
    source_emb: np.ndarray of shape (source_vocab_size, vec_dim)
    target_emb: np.ndarray of shape (target_vocab_size, vec_dim)
    dict_obj: np.ndarray of shape (dict_size, 2)
    orthogonal: bool

    Returns: np.ndarray of shape (vec_dim, vec_dim)
    """
    X_source = source_emb[dict_obj[:, 0]]  # shape (dict_size, vec_dim)
    X_target = target_emb[dict_obj[:, 1]]  # shape (dict_size, vec_dim)

    if orthogonal:
        u, s, vt = np.linalg.svd(np.dot(X_target.T, X_source))
        W_source = np.dot(vt.T, u.T)
    else:
        W_source = np.matmul(np.linalg.pinv(X_source), X_target)

    return W_source


def main(args):
    print(str(args))

    # load source and target embeddings
    if args.pickle:
        with open(args.source_embedding, 'rb') as fin:
            src_wv = pickle.load(fin)
        with open(args.target_embedding, 'rb') as fin:
            trg_wv = pickle.load(fin)
    else:
        src_wv = WordVecs(args.source_embedding, emb_format=args.format).normalize(args.normalize)
        trg_wv = WordVecs(args.target_embedding, emb_format=args.format).normalize(args.normalize)

    # load bilingual lexicon
    dict_obj = BilingualDict(args.gold_dictionary).get_indexed_dictionary(src_wv, trg_wv)

    # compute W_src and W_trg
    vec_dim = src_wv.vec_dim
    if args.project_source:
        W_src = get_W_source(src_wv.embedding, trg_wv.embedding, dict_obj, args.orthogonal)
        W_trg = np.identity(vec_dim, dtype=np.float32)
    else:
        W_src = np.identity(vec_dim, dtype=np.float32)
        W_trg = get_W_target(src_wv.embedding, trg_wv.embedding, dict_obj, args.orthogonal)

    save_model(W_src, W_trg, args.source_lang, args.target_lang, 'procruste', args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-bi', '--binary',
                        help='use 2-class set up',
                        action='store_true')
    parser.add_argument('-sl', '--source_lang',
                        help='source language: en/es/ca/eu (default: en)',
                        default='eu')
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
                        help='bilingual dictionary of source and target language (default: ./lexicons/bingliu/en-es.txt',
                        default='./lexicons/bingliu/en-es.txt')
    parser.add_argument('--project_source',
                        help='project source embedding (default: project target)',
                        action='store_true')
    parser.add_argument('--normalize',
                        choices=['unit', 'center', ],
                        nargs='*',
                        default=['center', 'unit'],
                        help='normalization actions')
    parser.add_argument('--orthogonal',
                        help='apply orthogonal restriction to the projection matrix',
                        action='store_true')
    parser.add_argument('--pickle',
                        action='store_true',
                        help='load from pickled wordvecs')
    parser.add_argument('--save_path',
                        default='checkpoints/artetxe.bin',
                        help='save path')

    lang_group = parser.add_mutually_exclusive_group()
    lang_group.add_argument('--en_es', action='store_true', help='train english-spanish embedding')
    lang_group.add_argument('--en_ca', action='store_true', help='train english-catalan embedding')
    lang_group.add_argument('--en_eu', action='store_true', help='train english-basque embedding')

    args = parser.parse_args()
    if args.en_es:
        src_emb_file = 'pickle/en.bin' if args.pickle else 'emb/wiki.en.vec'
        trg_emb_file = 'pickle/es.bin' if args.pickle else 'emb/wiki.es.vec'
        parser.set_defaults(source_lang='en', target_lang='es',
                            source_embedding=src_emb_file, target_embedding=trg_emb_file, format='fasttext_text',
                            gold_dictionary='lexicons/apertium/en-es.txt')
    elif args.en_ca:
        src_emb_file = 'pickle/en.bin' if args.pickle else 'emb/wiki.en.vec'
        trg_emb_file = 'pickle/ca.bin' if args.pickle else 'emb/wiki.ca.vec'
        parser.set_defaults(source_lang='en', target_lang='ca',
                            source_embedding=src_emb_file, target_embedding=trg_emb_file, format='fasttext_text',
                            gold_dictionary='lexicons/apertium/en-ca.txt')
    elif args.en_eu:
        src_emb_file = 'pickle/en.bin' if args.pickle else 'emb/wiki.en.vec'
        trg_emb_file = 'pickle/eu.bin' if args.pickle else 'emb/wiki.eu.vec'
        parser.set_defaults(source_lang='en', target_lang='eu',
                            source_embedding=src_emb_file, target_embedding=trg_emb_file, format='fasttext_text',
                            gold_dictionary='lexicons/apertium/en-eu.txt')

    args = parser.parse_args()
    main(args)
