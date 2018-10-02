"""
utils for loading binary Word2Vec, bilingual dictionary and OpeNER sentiment dataset. 

author: fyl
"""

import numpy as np
from scipy.spatial.distance import cosine
import csv
import collections
import sys
import os
from .cupy_utils import *


def length_normalize(X, inplace=True):
    """
    Normalize rows of X to unit length.

    X: np.ndarray (or cupy.ndarray)
    inplace: bool

    Returns: None or np.ndarray (or cupy.ndarray)
    """
    xp = get_array_module(X)
    norms = xp.sqrt(xp.sum(X**2, axis=1))
    norms[norms == 0.] = 1.
    if inplace:
        X /= norms[:, xp.newaxis]
    else:
        X = X / norms[:, xp.newaxis]
    return X


def mean_center(X, inplace=True):
    """
    X: np.ndarray (or cupy.ndarray)
    inplace: bool

    Returns: None or np.ndarray (or cupy.ndarray)
    """
    xp = get_array_module(X)
    if inplace:
        X -= xp.mean(X, axis=0)
    else:
        X = X - xp.mean(X, axis=0)
    return X


def normalize(X, actions, inplace=True):
    """
    X: np.ndarray (or cupy.ndarray)
    actions = list[str]
    inplace: bool

    Returns: None or np.ndarray (or cupy.ndarray)
    """
    for action in actions:
        if action == 'unit':
            X = length_normalize(X, inplace)
        elif action == 'center':
            X = mean_center(X, inplace)
    return X


def top_k_mean(X, k, inplace=False):
    """
    Average of top-k similarites.

    X: np.ndarray (or cupy.ndarray)
    k: int
    inplace: bool

    Returns: np.ndarray (or cupy.ndarray)
    """
    xp = get_array_module(X)
    size = X.shape[0]
    ans = xp.zeros(size, dtype=xp.float32)
    if k == 0:
        return ans
    if not inplace:
        X = X.copy()
    min_val = X.min()
    ind0 = xp.arange(size)
    ind1 = xp.zeros(size, dtype=xp.int32)
    for i in range(k):
        xp.argmax(X, axis=1, out=ind1)
        ans += X[ind0, ind1]
        X[ind0, ind1] = min_val
    ans /= k
    return ans


def dropout(X, keep_prob, inplace=True):
    """
    Randomly set entries of X to zeros.

    X: np.ndarray (or cupy.ndarray)
    keep_prob: float
    inplace: bool

    Returns: np.ndarray (or cupy.ndarray)
    """
    xp = get_array_module(X)
    mask = xp.random.rand(*X.shape) < keep_prob
    if inplace:
        X *= mask
    else:
        X = X * mask
    return X


def get_projection_matrix(X_src, X_trg, orthogonal, direction='forward', out=None):
    """
    X_src: ndarray
    X_trg: ndarray
    orthogonal: bool
    direction: str
        returns W_src if 'forward', W_trg otherwise
    """
    xp = get_array_module(X_src, X_trg)
    if orthogonal:
        if direction == 'forward':
            u, s, vt = xp.linalg.svd(xp.dot(X_trg.T, X_src))
            W = xp.dot(vt.T, u.T, out=out)
        elif direction == 'backward':
            u, s, vt = xp.linalg.svd(xp.dot(X_src.T, X_trg))
            W = xp.dot(vt.T, u.T, out=out)
    else:
        if direction == 'forward':
            W = xp.dot(xp.linalg.pinv(X_src), X_trg, out=out)
        elif direction == 'backward':
            W = xp.dot(xp.linalg.pinv(X_trg), X_src, out=out)
    return W


def get_unsupervised_init_dict(src_emb, trg_emb, cutoff_size, csls, norm_actions, direction):
    """
    Given source embedding and target embedding, return a initial bilingual
    dictionary base on similarity distribution.

    src_emb: ndarray of shape (src_size, vec_dim)
    trg_emb: ndarray of shape (trg_size, vec_dim)
    cutoff_size: int
    csls: int
    norm_actions: list[str]
    direction: str

    Returns: ndarray of shape (dict_size, 2)
    """
    xp = get_array_module(src_emb, trg_emb)
    sim_size = min(src_emb.shape[0], trg_emb.shape[0], cutoff_size) if cutoff_size > 0 else min(src_emb.shape[0], trg_emb.shape[0])
    u, s, vt = xp.linalg.svd(src_emb[:sim_size], full_matrices=False)
    src_sim = (u * s) @ u.T
    u, s, vt = xp.linalg.svd(trg_emb[:sim_size], full_matrices=False)
    trg_sim = (u * s) @ u.T
    del u, s, vt

    src_sim.sort(axis=1)
    trg_sim.sort(axis=1)
    normalize(src_sim, norm_actions)
    normalize(trg_sim, norm_actions)
    sim = xp.dot(src_sim, trg_sim.T)
    del src_sim, trg_sim
    src_knn_sim = top_k_mean(sim, csls, inplace=False)
    trg_knn_sim = top_k_mean(sim.T, csls, inplace=False)
    sim -= src_knn_sim[:, xp.newaxis] / 2 + trg_knn_sim / 2

    if direction == 'forward':
        init_dict = xp.stack([xp.arange(sim_size), xp.argmax(sim, axis=1)], axis=1)
    elif direction == 'backward':
        init_dict = xp.stack([xp.argmax(sim, axis=0), xp.arange(sim_size)], axis=1)
    elif direction == 'union':
        init_dict = xp.stack([xp.concatenate((xp.arange(sim_size), xp.argmax(sim, axis=0))), xp.concatenate((xp.argmax(sim, axis=1), xp.arange(sim_size)))], axis=1)
    return init_dict


class WordVecs(object):
    """
    Helper class for importing word embeddings in BINARY Word2Vec format.

    Parameters
    ----------
    file: str
        the binary embedding file containing word embbedings
    vocab: list[str] / set[str], optional (default None)
        if specified, out-of-vocabulary words won't be loaded
    encoding: str
        specify the encoding with which to decode the bytes
    normalize: bool
        mean center the word vectors and normalize to unit length
    """

    def __init__(self, file, vocab=None, encoding='utf-8', normalize=False):
        self.vocab = set(vocab) if vocab else None
        self.encoding = encoding
        self.vocab_size, self.vec_dim, self._matrix, self._w2idx, self._idx2w = self._read_vecs(
            file)
        self.vocab = set(self._w2idx.keys())
        if normalize:
            self.mean_center().normalize()

    def __getitem__(self, word):
        """
        Returns the vector representation of a word.

        word: str

        Returns: np.array
        """
        try:
            return self._matrix[self._w2idx[word]]
        except KeyError:
            raise KeyError('Word not in vocabulary')

    def _read_vecs(self, file):
        """
        Load word embeddings from the binary embedding file.

        file: str
        """
        with open(file, 'rb') as fin:
            header = fin.readline()

            vocab_size, vec_dim = map(int, header.split())
            bytes_each_word = np.dtype('float32').itemsize * vec_dim

            if self.vocab:
                vocab_size = len(self.vocab)

            emb_matrix = np.zeros((vocab_size, vec_dim), dtype='float32')
            w2idx = {}

            for _ in range(vocab_size):
                word = b''
                while True:
                    ch = fin.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':
                        word += ch
                word = word.decode(self.encoding)

                vec = np.fromstring(fin.read(bytes_each_word), dtype='float32')

                if self.vocab and word not in self.vocab:
                    continue
                else:
                    w2idx[word] = len(w2idx)
                    emb_matrix[w2idx[word]] = vec

        idx2w = [None] * len(w2idx)
        for w, i in w2idx.items():
            idx2w[i] = w
        idx2w = np.array(idx2w)
        return vocab_size, vec_dim, emb_matrix, w2idx, idx2w

    def add_word(self, word, vec):
        """
        Add a new word and its vector representation to the embedding matrix, then assign
        an index to this word.

        word: str
        vec: np.array of shape (vec_dim,)

        Returns: int
            index of the new word
        """
        if word in self._w2idx:
            raise ValueError('Word already in vocabulary')

        new_id = len(self._w2idx)

        if self._matrix.shape[0] == new_id:
            self._matrix = np.concatenate(
                (self._matrix, vec.reshape(1, self.vec_dim)), axis=0)
        else:
            self._matrix[new_id] = vec

        self._w2idx[word] = new_id
        self._idx2w = np.append(self._idx2w, word)
        return new_id

    def word2index(self, word):
        """
        Lookup the index of the word in the vocabulary.
        """
        try:
            return self._w2idx[word]
        except KeyError:
            raise KeyError('Word not in vocabulary')

    def index2word(self, index):
        """
        index: int / List[int] / np.ndarray of type int

        Returns: str / np.ndarray
        """
        try:
            return self._idx2w[index]
        except IndexError:
            raise IndexError('Invalid index')

    @property
    def embedding(self):
        return self._matrix

    def most_similar(self, word, num_similar=5):
        """
        Returns the k most similar word to the given word.

        word: str
            specify the word to be compared among the vocab
        num_similar: int, optional (default 5)
            specify k

        Returns: list[[int, str]]
            [[dist1, word1], [dist2, word2], ..., [dist_k, word_k]]
        """
        idx = self.word2index[word]
        vec = self._matrix[idx]
        most_similar = [(1, 0)] * num_similar
        for i, cur_vec in enumerate(self._matrix):
            if i == idx:
                continue
            dist = cosine(vec, cur_vec)
            if dist < most_similar[-1][0]:
                most_similar.pop()
                most_similar.append((dist, i))
                most_similar = sorted(most_similar)

        return [[dist, self._idx2w[i]] for dist, i in most_similar]

    def normalize(self, actions=None):
        if actions is None:
            norms = np.sqrt(np.sum(self._matrix**2, axis=1))
            norms[norms == .0] = 1
            self._matrix /= norms[:, np.newaxis]
        else:
            normalize(self._matrix, actions, inplace=True)
        return self

    def mean_center(self):
        avg = np.mean(self._matrix, axis=0)
        self._matrix -= avg
        return self


class BilingualDict(object):
    """
    Helper class for loading the bilingual dictionary. Each line in the ditionary
    file should contains two word separated by a tab.

    Parameters
    ----------
    infile: str
        the dictionary file
    """

    def __init__(self, infile):
        with open(infile, 'r', encoding='utf-8') as fin:
            csvin = csv.reader(fin, delimiter='\t')
            self.dictionary = [row for row in csvin if len(row) == 2]
            self._lookup = dict(self.dictionary)

    def __getitem__(self, word):
        """
        Given a word in the source language, returns its translation in target language.
        """
        try:
            return self._lookup[word]
        except KeyError:
            raise KeyError('Word not in dictionary')

    def filter(self, filter_func=lambda _: True):
        """
        filter_func: function object
            only those entries for which the function return True will be kept

        Retunrs: self
        """
        self.dictionary = [row for row in self.dictionary if filter_func(row)]
        return self

    def get_indexed_dictionary(self, source_wordvec, target_wordvec):
        """
        source_wordvec: WordVecs object
        target_wordvec: WordVecs object

        Returns: numpy.ndarray of shape (dicsize, 2)
        """
        dic_in_index = []
        for src_word, tgt_word in self.dictionary:
            try:
                src_id = source_wordvec.word2index(src_word)
                tgt_id = target_wordvec.word2index(tgt_word)
            except KeyError:
                continue
            dic_in_index.append([src_id, tgt_id])
        return np.array(dic_in_index)


class SentimentDataset(object):
    """
    Helper class for loading OpeNER datasets.

    Parameters
    ----------
    directory: str
        the directory of the datase
        <directory> --> train|dev|test --> pos|strpos|neg|strneg
    """

    def __init__(self, directory):
        self._load_dataset(directory)

    def _load_dataset(self, directory):
        """
        Given a base directory, load train/dev/test data.
        """
        def load(d):
            def load_category(category):
                with open(os.path.join(d, category), 'r', encoding='utf-8') as fin:
                    sents = fin.readlines()
                return [row.split() for row in sents]

            ans = [load_category(cate)
                   for cate in ['pos.txt', 'strpos.txt', 'neg.txt', 'strneg.txt']]
            X = sum(ans, [])
            y = np.concatenate([np.full(len(t), i)
                                for i, t in enumerate(ans)], axis=0)
            return X, y

        self.train = load(os.path.join(directory, 'train'))
        self.dev = load(os.path.join(directory, 'dev'))
        self.test = load(os.path.join(directory, 'test'))

    def to_index(self, wordvecs):
        """
        wordvecs: WordVecs object

        Returns: self
        """
        def sents2index(X, y):
            X_new = []
            for sent in X:
                sent_new = []
                for word in sent:
                    try:
                        sent_new.append(wordvecs.word2index(word.lower()))
                    except KeyError:
                        continue
                X_new.append(sent_new)
            return X_new, y

        self.train = sents2index(*self.train)
        self.dev = sents2index(*self.dev)
        self.test = sents2index(*self.test)
        return self

    def to_vecs(self, emb):
        """
        emb: ndarray of shape (vocab_size, vec_dim)

        Returns: self
        """
        def ind2vec(X, y):
            size = len(X)
            vec_dim = emb.shape[1]
            X_new = np.zeros((size, vec_dim), dtype=np.float32)
            for i, row in enumerate(X):
                if len(row) > 0:
                    X_new[i] = np.mean(emb[row], axis=0)
            return X_new, y

        self.train = ind2vec(*self.train)
        self.dev = ind2vec(*self.dev)
        self.test = ind2vec(*self.test)
        return self


class SentiWordSet(object):
    """
    Helper class for loading sentimental words for further exmanation.

    Parameters
    ----------
    path: str
        file location of the word set
    encoding: str
        the encoding method of the file
    """

    def __init__(self, path, encoding='utf-8'):
        self.labels, self.wordsets = self._load_words(path, encoding)

    def _load_words(self, path, encoding):
        labels, wordsets = [], []
        with open(path, 'r', encoding=encoding) as fin:
            for line in fin:
                category, words = line.split(' :: ')
                labels.append(category)
                wordsets.append(words.split())
        return labels, wordsets

    def to_index(self, wordvecs):
        """
        Given a WordVecs object, convert words to indices.

        Returns: self
        """
        for i, words in enumerate(self.wordsets):
            indices = []
            for w in words:
                try:
                    indices.append(wordvecs.word2index(w))
                except KeyError:
                    pass
            self.wordsets[i] = indices
        return self


class BDI(object):
    """
    Helper class for bilingual dictionary induction.

    Parameters
    ----------
    src_emb: np.ndarray of shape (src_emb_size, vec_dim)
    trg_emb: np.ndarray of shape (trg_emb_size, vec_dim)
    batch_size: int
    cuda: bool
    """

    def __init__(self, src_emb, trg_emb, batch_size=5000, cutoff_size=10000, cutoff_type='both', direction=None, csls=10, batch_size_val=1000):
        if cutoff_type == 'oneway' and csls > 0:
            raise ValueEror("cutoff_type='both' and csls > 0 not supported")  # TODO

        xp = get_array_module(src_emb, trg_emb)
        self.xp = xp
        self.src_emb = src_emb
        self.trg_emb = trg_emb
        self.batch_size = batch_size
        self.cutoff_size = cutoff_size
        self.cutoff_type = cutoff_type
        self.direction = direction
        self.csls = csls
        self.batch_size_val = batch_size_val
        self.src_proj_emb = self.src_emb[:self.cutoff_size].copy()
        self.trg_proj_emb = self.trg_emb.copy()

        self.src_size = src_emb.shape[0]
        self.trg_size = trg_emb.shape[0]
        if direction in ('forward', 'union') or csls > 0:
            self.fwd_src_size = cutoff_size
            self.fwd_trg_size = cutoff_size if cutoff_type == 'both' else trg_size
            self.fwd_ind = xp.arange(self.fwd_src_size, dtype=xp.int32)
            self.fwd_trg = xp.empty(self.fwd_src_size, dtype=xp.int32)
            self.fwd_sim = xp.empty((batch_size, self.fwd_trg_size), dtype=xp.float32)
        if direction in ('backward', 'union') or csls > 0:
            self.bwd_trg_size = cutoff_size
            self.bwd_src_size = cutoff_size if cutoff_type == 'both' else src_size
            self.bwd_ind = xp.arange(self.bwd_trg_size, dtype=xp.int32)
            self.bwd_src = xp.arange(self.bwd_trg_size, dtype=xp.int32)
            self.bwd_sim = xp.empty((batch_size, self.bwd_src_size), dtype=xp.float32)
        self.sim_val = xp.empty((batch_size_val, self.trg_size), dtype=xp.float32)
        self.dict_size = cutoff_size * 2 if direction == 'union' else cutoff_size
        self.dict = xp.empty((self.dict_size, 2), dtype=xp.int32)

        if csls > 0:
            if direction in ('forward', 'union'):
                self.fwd_knn_sim = xp.empty(self.fwd_trg_size, dtype=xp.float32)
            if direction in ('backward', 'union'):
                self.bwd_knn_sim = xp.empty(self.bwd_src_size, dtype=xp.float32)

    def project(self, W, direction='backward'):
        """
        W_target: ndarray of shape (vec_dim, vec_dim)

        Returns: self
        """
        xp = self.xp
        if direction == 'forward':
            xp.dot(self.src_emb[:self.cutoff_size], W, out=self.src_proj_emb)
        else:
            xp.dot(self.trg_emb, W, out=self.trg_proj_emb)
        return self

    def get_bilingual_dict_with_cutoff(self, keep_prob=1.):
        """
        keep_prob: float

        Returns: ndarray of shape (dict_size, 2)
        """
        xp = self.xp
        if self.direction in ('forward', 'union'):
            if self.csls > 0:
                for i in range(0, self.fwd_trg_size, self.batch_size):
                    j = min(self.fwd_trg_size, i + self.batch_size)
                    xp.dot(self.trg_proj_emb[i:j], self.src_proj_emb[:self.fwd_src_size].T, out=self.bwd_sim[:j - i])
                    self.fwd_knn_sim[i:j] = top_k_mean(self.bwd_sim[:j - i], self.csls, inplace=True)
            for i in range(0, self.fwd_src_size, self.batch_size):
                j = min(self.fwd_src_size, i + self.batch_size)
                xp.dot(self.src_proj_emb[i:j], self.trg_proj_emb[:self.fwd_trg_size].T, out=self.fwd_sim[:j - i])
                self.fwd_sim[:j - i] -= self.fwd_knn_sim / 2
                dropout(self.fwd_sim[:j - i], keep_prob, inplace=True).argmax(axis=1, out=self.fwd_trg[i:j])
        if self.direction in ('backward', 'union'):
            if self.csls > 0:
                for i in range(0, self.bwd_src_size, self.batch_size):
                    j = min(self.bwd_src_size, i + self.batch_size)
                    xp.dot(self.src_proj_emb[i:j], self.trg_proj_emb[:self.bwd_trg_size].T, out=self.fwd_sim[:j - i])
                    self.bwd_knn_sim[i:j] = top_k_mean(self.fwd_sim[:j - i], self.csls, inplace=True)
            for i in range(0, self.bwd_trg_size, self.batch_size):
                j = min(self.bwd_trg_size, i + self.batch_size)
                xp.dot(self.trg_proj_emb[i:j], self.src_proj_emb[:self.bwd_src_size].T, out=self.bwd_sim[:j - i])
                self.bwd_sim[:j - i] -= self.bwd_knn_sim / 2
                dropout(self.bwd_sim[:j - i], keep_prob, inplace=True).argmax(axis=1, out=self.bwd_src[i:j])
        if self.direction == 'forward':
            xp.stack([self.fwd_ind, self.fwd_trg], axis=1, out=self.dict)
        elif self.direction == 'backward':
            xp.stack([self.bwd_src, self.bwd_ind], axis=1, out=self.dict)
        elif self.direction == 'union':
            self.dict[:, 0] = xp.concatenate((self.fwd_ind, self.bwd_src))
            self.dict[:, 1] = xp.concatenate((self.fwd_trg, self.bwd_ind))
        return self.dict.copy()

    def get_target_indices(self, src_ind):
        """
        src_ind: np.ndarray of shape (dict_size,)

        Returns: np.ndarray of shape (dict_size,)
        """
        xp = self.xp
        size = src_ind.shape[0]
        trg_ind = xp.empty(size, dtype=xp.int32)
        for i in range(0, size, self.batch_size_val):
            j = min(i + self.batch_size_val, size)
            xp.dot(self.src_proj_emb[src_ind[i:j]], self.trg_proj_emb.T, out=self.sim_val[: j - i])
            xp.argmax(self.sim_val[:j - i], axis=1, out=trg_ind[i:j])
        return trg_ind
