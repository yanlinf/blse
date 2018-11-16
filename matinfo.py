import numpy as np
import argparse
from utils.model import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infiles',
                        nargs='+',
                        help='W_src and W_trg')

    args = parser.parse_args()
    for infile in args.infiles:
        dic = load_model(infile)
        u, s1, vt = np.linalg.svd(dic['W_source'])
        u, s2, vt = np.linalg.svd(dic['W_target'])
        print('-------------------------------------------------')
        print('file: {}'.format(infile))
        print('source mean singular value: {0:.4f}'.format(float(s1.mean())))
        print('source top 6 singular values: {0}'.format(str(s1[:6])))
        print('target mean singular value: {0:.4f}'.format(float(s2.mean())))
        print('target top 6 singular values: {0}'.format(str(s2[:6])))


if __name__ == '__main__':
    main()
