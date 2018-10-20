import pickle
import os
from utils.dataset import *
from utils.math import *
from utils.bdi import *

TARGET = 'emb/wiki.%s.vec'
FORMAT = 'fasttext_text'


def main():
    if not os.path.exists('pickle'):
        os.mkdir('pickle')
    for lang in ('eu', 'en', 'es', 'ca'):
        wv = WordVecs(TARGET % lang, emb_format=FORMAT).normalize(['center', 'unit'])
        with open('pickle/%s.bin' % lang, 'wb') as fout:
            pickle.dump(wv, fout)


if __name__ == '__main__':
    main()
