"""
helpers for loading and saving models.

author: fyl
"""
import pickle
import os

def save_model(W_src, W_trg, model_type, path):
    dic = {
        "W_source": W_src,
        "W_target": W_trg,
        "model": model_type,
    }
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.mkdir(directory)
    with open(path, 'wb') as fout:
        pickle.dump(dic, fout)


def load_model(path):
    with open(path, 'rb') as fout:
        dic = pickle.dump(dic, fout)
    return dic['W_source'], dic['W_target'], dic['model']
