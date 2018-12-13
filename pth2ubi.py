import torch
import numpy as np
import argparse
import pickle
import os
from utils.model import *


def main(args):
    for infile in args.infile:
        W_src = torch.load(infile).astype(np.float32)
        W_trg = np.identity(W_src.shape[0], dtype=np.float32)
        src_lang = 'en'
        if 'en-es' in infile:
            trg_lang = 'es'
            i = infile.find('en-es')
            j = infile.find('/', i)
            idx = infile[i:j]
        elif 'en-ca' in infile:
            trg_lang = 'ca'
            i = infile.find('en-ca')
            j = infile.find('/', i)
            idx = infile[i:j]
        elif 'en-eu' in infile:
            trg_lang = 'eu'
            i = infile.find('en-eu')
            j = infile.find('/', i)
            idx = infile[i:j]
        savepath = 'checkpoints/en-{}-adv-ubi-{}.bin'.format(trg_lang, idx)
        save_model(W_src.T.copy(), W_trg, src_lang, trg_lang, 'ubi', savepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile',
                        nargs='+',
                        help='checkpoints files')
    args = parser.parse_args()
    main(args)
