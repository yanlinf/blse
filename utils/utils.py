"""
utils for loading binary Word2Vec, bilingual dictionary and OpeNER sentiment dataset. 

author: fyl
"""

import numpy as np
from scipy.spatial.distance import cosine
import csv
import collections
import os


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
    """

    def __init__(self, file, vocab=None, encoding='utf-8'):
        self.vocab = set(vocab) if vocab else None
        self.encoding = encoding
        self.vocab_size, self.vec_dim, self._matrix, self._w2idx, self._idx2w = self._read_vecs(
            file)

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

    def normalize(self):
        norms = np.sqrt(np.sum(self._matrix**2, axis=1))
        norms[norms == .0] = 1
        self._matrix /= norms[:, np.newaxis]

    def mean_center(self):
        avg = np.mean(self._matrix, axis=0)
        self._matrix -= avg


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

        Returns: None
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


class SentiWordSet(object):
    """
    Helper class for loading sentimental words for further exmanation.

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
