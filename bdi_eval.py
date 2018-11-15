import argparse
import pickle
import numpy as np
from utils.dataset import *
from utils.math import *
from utils.bdi import *
from utils.cupy_utils import *
from utils.model import *


def main(args):
    print(str(args))

    for infile in args.W:
        dic = load_model(infile)
        W_src = dic['W_source']
        W_trg = dic['W_target']
        src_lang = dic['source_lang']
        trg_lang = dic['target_lang']
        model = dic['model']
        with open('pickle/%s.bin' % src_lang, 'rb') as fin:
            src_wv = pickle.load(fin)
        with open('pickle/%s.bin' % trg_lang, 'rb') as fin:
            trg_wv = pickle.load(fin)

        if trg_lang in ('es', 'ca', 'eu'):
            gold_dict = BilingualDict('lexicons/apertium/{}-{}.txt'.format(src_lang, trg_lang)).get_indexed_dictionary(src_wv, trg_wv)
        else:
            gold_dict = BilingualDict('lexicons/muse/{}-{}.0-5000.txt'.format(src_lang, trg_lang)).get_indexed_dictionary(src_wv, trg_wv)

        unit_norm = model in ('ubise',)
        xw = xp.array(src_wv.embedding[gold_dict[:, 0]].dot(W_src))
        zw = xp.array(trg_wv.embedding.dot(W_trg))
        if unit_norm:
            length_normalize(xw, inplace=True)
            length_normalize(zw, inplace=True)

        m = xw.shape[0]
        vbs = args.val_batch_size
        s = xp.empty((vbs, zw.shape[0]), dtype=xp.float32)
        tidx = xp.empty(m, dtype=xp.int32)
        if args.scorer in ('euclidean',):
            t = l2norm(zw)**2
        for i in range(0, m, vbs):
            j = min(m, i + vbs)
            xw[i:j].dot(zw.T, out=s[:j - i])
            if args.scorer in ('euclidean',):
                s[:j - i] -= t / 2
            xp.argmax(s[:j - i], axis=1, out=tidx[i:j])

        accuracy = (asnumpy(tidx) == gold_dict[:, 1]).mean()
        print('file: {}   acc: {}'.format(infile, accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('W',
                        nargs='+',
                        default=['checkpoints/senti.bin'],
                        help='W_src and W_trg')
    parser.add_argument('--scorer',
                        choices=['euclidean', 'dot'],
                        default='dot',
                        help='retrieval method')
    parser.add_argument('-vbs',
                        '--val_batch_size',
                        default=300,
                        type=int,
                        help='training batch size (default: 300)')
    parser.add_argument('--cuda',
                        action='store_true',
                        help='use cuda to accelerate')

    args = parser.parse_args()

    if args.cuda:
        xp = get_cupy()
        if xp is None:
            print('Install cupy for cuda support')
            sys.exit(-1)
    else:
        xp = np

    main(args)
