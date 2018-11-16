import tensorflow as tf
import numpy as np
import argparse
from sklearn.metrics import f1_score
import pickle
from utils.dataset import *
from utils.math import *
from utils.bdi import *
import logging


class AttenAverage(object):
    """
    Self attention for sentiment classification.
    """

    def __init__(self, sess, vec_dim, nclasses, learning_rate, batch_size, num_epoch, num_atten=4, pad=256):
        self.sess = sess
        self.vec_dim = vec_dim
        self.nclasses = nclasses
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.num_atten = num_atten
        self.pad = pad
        self._build_graph()
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.pad, self.vec_dim))
        self.labels = tf.placeholder(tf.int32, shape=(None,))

        W1 = tf.get_variable('W1', (self.vec_dim, self.num_atten), tf.float32, initializer=tf.random_uniform_initializer(-1., 1.))
        b1 = tf.get_variable('b1', (self.num_atten), tf.float32, initializer=tf.zeros_initializer())
        atten = tf.reshape(self.inputs, (-1, self.vec_dim)) @ W1 + b1
        atten_norm = tf.nn.softmax(tf.reduce_max(tf.reshape(atten, (-1, self.pad, self.num_atten)), axis=-1), axis=-1)  # shape (batch_size, self.pad)
        self.atten_norm = atten_norm

        L1 = tf.expand_dims(atten_norm, axis=-1) * self.inputs  # shape (batch_size, self.pad, 300)
        L1 = tf.reduce_sum(L1, axis=1)  # shape (batch_size, 300)

        self.L1 = L1

        W2 = tf.get_variable('W2', (self.vec_dim, self.nclasses), tf.float32, initializer=tf.random_uniform_initializer(-1., 1.))
        b2 = tf.get_variable('b2', (self.nclasses), tf.float32, initializer=tf.zeros_initializer())

        logits = L1 @ W2 + b2

        self.pred = tf.argmax(logits, axis=1)
        self.loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.labels, self.nclasses), logits)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def fit(self, train_x, train_y, test_x=None, test_y=None):
        nsample = len(train_x)
        for epoch in range(self.num_epoch):
            loss = 0.
            pred = np.zeros(nsample)
            for index, offset in enumerate(range(0, nsample, self.batch_size)):
                xs = train_x[offset:offset + self.batch_size]
                ys = train_y[offset:offset + self.batch_size]
                _, loss_, pred_, = self.sess.run([self.optimizer, self.loss, self.pred],
                                                 {self.inputs: xs, self.labels: ys})
                loss += loss_ * len(xs)
                pred[offset:offset + self.batch_size] = pred_
            loss /= nsample
            fscore = f1_score(train_y, pred, average='macro')

            if test_x is not None and test_y is not None:
                print('\repoch: {}   f1: {:.4f}   loss: {:.6f}   test_f1: {:.4f}'.format(epoch, fscore, loss, self.score(test_x, test_y)), end='')
            else:
                print('\repoch: {}   f1: {:.4f}   loss: {:.6f}   test_f1: {:.4f}'.format(epoch, fscore, loss), end='')
        print()

    def predict(self, test_x):
        pred = self.sess.run(self.pred, {self.inputs: test_x})
        return pred

    def score(self, test_x, test_y, scorer='f1_macro'):
        if scorer == 'f1_macro':
            return f1_score(test_y, self.predict(test_x), average='macro')
        else:
            raise NotImplementedError()

    def predict_attention_scores(self, test_x):
        atten_ = self.sess.run(self.atten_norm, {self.inputs: test_x})
        return atten_

    def get_senti_x(self, X):
        return self.sess.run(self.L1, {self.inputs: X})


def make_data(X, y, embedding, vec_dim, binary, pad_id, shuffle=True):
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=args.pad, padding='post', value=pad_id)
    X = embedding[X]  # shape (nsamples, args.pad, vec_dim)
    if shuffle:
        perm = np.random.permutation(X.shape[0])
        X, y = X[perm], y[perm]
    if binary:
        y = (y >= 2).astype(np.int32)
    return X, y


def print_examples_with_attention(X, y, pred, wordvecs, attention):
    print('---------------------------------------------------------')
    for sent, label, pred_label, atten in zip(X, y, pred, attention):
        atten = atten[:len(sent)]
        print([(wordvecs.index2word(j), a) for j, a in zip(sent, atten)])
        print('true label:', label, '   predicted label: ', pred_label)
        print('---------------------------------------------------------')


def main(args):
    logging.info(str(args))
    with open(args.source_embedding, 'rb') as fin:
        src_wv = pickle.load(fin)
    vec_dim = src_wv.embedding.shape[1]
    src_pad_id = src_wv.add_word('<PAD>', np.zeros(vec_dim))
    src_ds = SentimentDataset(args.source_dataset).to_index(src_wv)
    train_x, train_y = make_data(*src_ds.train, src_wv.embedding, vec_dim, args.binary, src_pad_id)
    test_x, test_y = make_data(*src_ds.test, src_wv.embedding, vec_dim, args.binary, src_pad_id)
    with tf.Session() as sess:
        model = AttenAverage(sess, vec_dim, (2 if args.binary else 4),
                             args.learning_rate, args.batch_size, args.epochs, pad=args.pad)
        model.fit(train_x, train_y, test_x, test_y)
        logging.info('Test f1_macro: %.4f' % model.score(test_x, test_y))

        train_x, train_y = make_data(*src_ds.train, src_wv.embedding, vec_dim, args.binary, src_pad_id, shuffle=False)
        test_x, test_y = make_data(*src_ds.test, src_wv.embedding, vec_dim, args.binary, src_pad_id, shuffle=False)

        xsenti = model.get_senti_x(train_x)
        ysenti = train_y * 2 if args.binary else train_y
        with open(args.save_path, 'wb') as fout:
            pickle.dump((xsenti, ysenti), fout)

        # print_examples_with_attention(src_ds.test[0][:50], src_ds.test[1][:50],
        #                   model.predict(test_x[:50]), src_wv, model.predict_attention_scores(test_x[:50]))
        # print_examples_with_attention(src_ds.train[0][:50], src_ds.train[1][:50],
        #                   model.predict(train_x[:50]), src_wv, model.predict_attention_scores(train_x[:50]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-bi', '--binary',
                        help='use 2-class set up',
                        action='store_true')
    parser.add_argument('-lr', '--learning_rate',
                        help='learning rate (default: 0.001)',
                        type=float,
                        default=0.001)
    parser.add_argument('-e', '--epochs',
                        help='training epochs (default: 200)',
                        default=200,
                        type=int)
    parser.add_argument('-bs', '--batch_size',
                        help='training batch size (default: 50)',
                        default=50,
                        type=int)
    parser.add_argument('-se', '--source_embedding',
                        help='monolingual word embedding of the source language (default: ./emb/en.bin)',
                        default='./pickle/en.bin')
    parser.add_argument('-sd', '--source_dataset',
                        help='sentiment dataset of the source language',
                        default='./datasets/en/opener_sents/')
    parser.add_argument('--save_path',
                        default='pickle/senti.bin',
                        help='save path')
    parser.add_argument('--pad',
                        type=int,
                        default=256,
                        help='padding size')
    parser.add_argument('--debug',
                        help='print debug info',
                        action='store_const',
                        dest='loglevel',
                        default=logging.INFO,
                        const=logging.DEBUG)

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel,
                        format='%(asctime)s: %(levelname)s: %(message)s')
    main(args)
