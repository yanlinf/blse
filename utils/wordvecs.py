import numpy as np
import logging
from scipy.spatial.distance import cosine


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

        idx2w = {i: w for w, i in w2idx.items()}
        return vocab_size, vec_dim, emb_matrix, w2idx, idx2w

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
        idx = self._w2idx[word]
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
