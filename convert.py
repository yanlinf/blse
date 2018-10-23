import argparse
import pickle
from utils.model import *


def main(args):
    for infile in args.infile:
        dic = load_model(infile)
        dic['model'] = args.model
        W_src = dic.pop('W_source')
        W_trg = dic.pop('W_target')
        source_lang = dic.pop('source_lang')
        target_lang = dic.pop('target_lang')
        model_type = dic.pop('model')
        if '.bin' in infile:
            savepath = infile.replace('.bin', '-{0}.bin'.format(args.model))
        else:
            savepath = infile + '-' + args.model
        save_model(W_src, W_trg, source_lang, target_lang, model_type, savepath, **dic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile',
                        nargs='+',
                        help='checkpoints files')
    parser.add_argument('model',
                        help='target model type')
    args = parser.parse_args()
    main(args)
