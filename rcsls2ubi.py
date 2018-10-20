import numpy as np
import argparse
import pickle
import os
from utils.model import *


def main(args):
    for infile in args.infile:
        with open(infile, 'r') as fin:
            m, n = map(int, fin.readline().rstrip().split())
            if m != n:
                raise ValueError('m not equal to n')
            W_src = np.empty((m, n), dtype=np.float32)
            for i in range(m):
                W_src[i] = np.fromstring(fin.readline(), sep=' ', dtype=np.float32)
        W_trg = np.identity(m, dtype=np.float32)
        src_lang, trg_lang, _ = os.path.basename(infile).split('-', 2)
        if '.bin' in infile:
            savepath = infile.replace('.bin', '-ubi.bin')
        else:
            savepath = infile + '-ubi'
        save_model(W_src.T.copy(), W_trg, src_lang, trg_lang, 'ubi', savepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile',
                        nargs='+',
                        help='checkpoints files')
    args = parser.parse_args()
    main(args)
