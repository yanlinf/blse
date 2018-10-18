import pickle
import os
import numpy as np
from utils.dataset import *
from utils.math import *
from utils.bdi import *

TARGET = 'checkpoints/en-%s-identity'


def main():
    if not os.path.exists('pickle'):
        os.mkdir('pickle')
    for lang in ('eu', 'es', 'ca'):
        dic = {
            'W_source': np.identity(300, dtype=np.float32),
            'W_target': np.identity(300, dtype=np.float32),
            'source_lang': 'en',
            'target_lang': lang,
            'model': 'ubi',
        }
        with open(TARGET % lang, 'wb') as fout:
            pickle.dump(dic, fout)


if __name__ == '__main__':
    main()
