import pickle
from utils.dataset import *
from utils.math import *
from utils.bdi import *


def main():
    for lang in ('eu', 'en', 'es', 'ca'):
        wv = WordVecs('emb/wiki.%s.vec' % lang, emb_format='fasttext_text').normalize(['center', 'unit'])
        with open('pickle/%s.bin' % lang, 'wb') as fout:
            pickle.dump(wv, fout)

    
if __name__ == '__main__':
    main()