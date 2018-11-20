import tensorflow as tf
import numpy as np
import argparse
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from utils.dataset import *
from utils.math import *
from utils.bdi import *
from utils.model import *
import glob
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

MAX_LEN = 64
TMP_FILE = 'tmp/cnn{:d}.ckpt'.format(np.random.randint(0, 1e10))


class SentiCNN(object):
    """
    CNN for sentiment classification.
    """

    def __init__(self, sess,
                 vec_dim=300,
                 nclasses=2,
                 learning_rate=0.001,
                 batch_size=50,
                 num_epoch=200,
                 num_filters=64,
                 dropout=0.5,
                 C=0.03,
                 clip=False):
        self.sess = sess
        self.vec_dim = vec_dim
        self.nclasses = nclasses
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.num_filters = num_filters
        self.dropout = dropout
        self.C = C
        self.clip = clip
        self._build_graph()
        self.initialize()
        self.saver = tf.train.Saver()

    def _build_graph(self):
        self.keep_prob = tf.placeholder(tf.float32)
        self.inputs = tf.placeholder(tf.float32, shape=(None, MAX_LEN * self.vec_dim, 1))
        self.batch_weights = tf.placeholder(tf.float32, shape=(None,))
        self.labels = tf.placeholder(tf.int32, shape=(None,))

        W1 = tf.get_variable('W1', (self.vec_dim, 1, self.num_filters), tf.float32, initializer=tf.random_uniform_initializer(-1., 1.))
        W2 = tf.get_variable('W2', (2 * self.vec_dim, 1, self.num_filters), tf.float32, initializer=tf.random_uniform_initializer(-1., 1.))
        W3 = tf.get_variable('W3', (3 * self.vec_dim, 1, self.num_filters), tf.float32, initializer=tf.random_uniform_initializer(-1., 1.))
        b1 = tf.get_variable('b1', (self.num_filters), tf.float32, initializer=tf.zeros_initializer())
        b2 = tf.get_variable('b2', (self.num_filters), tf.float32, initializer=tf.zeros_initializer())
        b3 = tf.get_variable('b3', (self.num_filters), tf.float32, initializer=tf.zeros_initializer())

        conv1 = tf.nn.conv1d(self.inputs, W1, self.vec_dim, 'VALID') + b1  # shape (batch_size, length, nchannels)
        conv2 = tf.nn.conv1d(self.inputs, W2, self.vec_dim, 'VALID') + b2  # shape (batch_size, length, nchannels)
        conv3 = tf.nn.conv1d(self.inputs, W3, self.vec_dim, 'VALID') + b3  # shape (batch_size, length, nchannels)

        relu1 = tf.nn.relu(conv1)
        relu2 = tf.nn.relu(conv2)
        relu3 = tf.nn.relu(conv3)

        pool1 = tf.nn.dropout(tf.reduce_max(relu1, axis=1), keep_prob=self.keep_prob)  # shape (batch_size, nchannels)
        pool2 = tf.nn.dropout(tf.reduce_max(relu2, axis=1), keep_prob=self.keep_prob)  # shape (batch_size, nchannels)
        pool3 = tf.nn.dropout(tf.reduce_max(relu3, axis=1), keep_prob=self.keep_prob)  # shape (batch_size, nchannels)
        pooled = tf.concat((pool1, pool2, pool3), axis=1)
        self.relu1 = relu1

        W4 = tf.get_variable('W4', (3 * self.num_filters, self.nclasses), tf.float32, initializer=tf.random_uniform_initializer(-1., 1.))
        b4 = tf.get_variable('b4', (self.nclasses,), tf.float32, initializer=tf.zeros_initializer())
        logits = tf.matmul(pooled, W4) + b4

        params = [W1, W2, W3, W4]
        self.assigns = []
        for param in params:
            scale = tf.minimum(self.C / tf.sqrt(tf.reduce_sum(tf.square(param), axis=0)), 1)
            self.assigns.append(tf.assign(param, scale * param))

        self.W1 = W1
        self.W2 = W2
        self.b2 = b2
        self.pred = tf.argmax(logits, axis=1)
        self.loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.labels, self.nclasses), logits, weights=self.batch_weights)
        if not self.clip:
            self.loss = self.loss + (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4)) * self.C
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def initialize(self):
        self.sess.run(tf.global_variables_initializer())
        if self.clip:
            self.sess.run(self.assigns)

    def fit(self, train_x, train_y, dev_x=None, dev_y=None, weights=None):
        max_f1 = 0
        self.saver.save(self.sess, TMP_FILE)
        self.best_W1_ = self.best_W2_ = self.best_b2_ = None
        nsample = len(train_x)
        if weights is None:
            weights = np.ones(train_x.shape[0])
        for epoch in range(self.num_epoch):
            loss = 0.
            pred = np.zeros(nsample)
            for index, offset in enumerate(range(0, nsample, self.batch_size)):
                xs = train_x[offset:offset + self.batch_size]
                ys = train_y[offset:offset + self.batch_size]
                ws = weights[offset:offset + self.batch_size]
                _, loss_, pred_, relu1_ = self.sess.run([self.optimizer, self.loss, self.pred, self.relu1],
                                                        {self.inputs: xs,
                                                         self.labels: ys,
                                                         self.keep_prob: self.dropout,
                                                         self.batch_weights: ws})
                if self.clip:
                    self.sess.run(self.assigns)
                loss += loss_ * len(xs)
                pred[offset:offset + self.batch_size] = pred_
            loss /= nsample
            fscore = f1_score(train_y, pred, average='macro')

            if dev_x is not None and dev_y is not None:
                dev_f1 = self.score(dev_x, dev_y)
                print('epoch: {:d}  f1: {:.4f}  loss: {:.6f}  dev_f1: {:.4f}\r'.format(epoch, fscore, loss, dev_f1), end='', flush=True)
                if dev_f1 > max_f1:
                    max_f1 = dev_f1
                    self.best_W1_, self.best_W2_, self.best_b2_ = self.sess.run([self.W1, self.W2, self.b2])
                    self.saver.save(self.sess, TMP_FILE)
            else:
                print('epoch: {:d}  f1: {:.4f}  loss: {:.6f}\r'.format(epoch, fscore, loss), end='', flush=True)
        print()
        self.best_score_ = max_f1
        if dev_x is None or dev_y is None:
            self.best_W1_, self.best_W2_, self.best_b2_ = self.sess.run([self.W1, self.W2, self.b2])
        else:
            self.saver.restore(self.sess, TMP_FILE)

    def predict(self, test_x):
        pred = self.sess.run(self.pred, {self.inputs: test_x, self.keep_prob: 1.})
        return pred

    def score(self, test_x, test_y, scorer='f1_macro'):
        if scorer == 'f1_macro':
            return f1_score(test_y, self.predict(test_x), average='macro')
        else:
            raise NotImplementedError()

    def save(self, savepath):
        self.saver.save(self.sess, savepath)


def main(args):
    print(str(args))

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4

    if args.output is not None:
        with open(args.output, 'w', encoding='utf-8') as fout:
            fout.write('infile,src_lang,trg_lang,model,is_binary,{},f1_macro,best_dev\n'.format(','.join(['f1_{:.4f}'.format(C) for C in args.C])))

    if args.setting == 'both':
        settings = [True, False]
    elif args.setting == 'binary':
        settings = [True, ]
    elif args.setting == '4-class':
        settings = [False, ]

    for infile in args.W:
        dic = load_model(infile)
        W_src = dic['W_source']
        W_trg = dic['W_target']
        src_lang = dic['source_lang']
        trg_lang = dic['target_lang']
        model = dic['model']
        with open('pickle/%s.bin' % src_lang, 'rb') as fin:
            src_wv = pickle.load(fin)
        with open('pickle/%s.bin' % trg_lang, 'rb') as fin:
            trg_wv = pickle.load(fin)
        src_pad_id = src_wv.add_word('<pad>', np.zeros(src_wv.vec_dim, dtype=np.float32))
        trg_pad_id = trg_wv.add_word('<pad>', np.zeros(trg_wv.vec_dim, dtype=np.float32))
        src_proj_emb = np.empty(src_wv.embedding.shape, dtype=np.float32)
        trg_proj_emb = np.empty(trg_wv.embedding.shape, dtype=np.float32)
        if model == 'ubise':
            src_wv.embedding.dot(W_src, out=src_proj_emb)
            length_normalize(src_proj_emb, inplace=True)
            trg_wv.embedding.dot(W_trg, out=trg_proj_emb)
            length_normalize(trg_proj_emb, inplace=True)
        elif model == 'ubi':
            src_wv.embedding.dot(W_src, out=src_proj_emb)
            trg_wv.embedding.dot(W_trg, out=trg_proj_emb)
        elif model == 'blse':
            src_wv.embedding.dot(W_src, out=src_proj_emb)
            trg_wv.embedding.dot(W_trg, out=trg_proj_emb)
        else:
            src_wv.embedding.dot(W_src, out=src_proj_emb)
            trg_wv.embedding.dot(W_trg, out=trg_proj_emb)

        for is_binary in settings:
            src_ds = SentimentDataset('datasets/%s/opener_sents/' % src_lang).to_index(src_wv, binary=is_binary).pad(src_pad_id, MAX_LEN).to_vecs(src_proj_emb, True)
            trg_ds = SentimentDataset('datasets/%s/opener_sents/' % trg_lang).to_index(trg_wv, binary=is_binary).pad(trg_pad_id, MAX_LEN).to_vecs(trg_proj_emb, True)
            vec_dim = src_proj_emb.shape[1]

            train_x = src_ds.train[0].reshape((-1, MAX_LEN * vec_dim, 1))
            train_y = src_ds.train[1]
            dev_x = np.concatenate((trg_ds.train[0], trg_ds.dev[0]), axis=0).reshape((-1, MAX_LEN * vec_dim, 1))
            dev_y = np.concatenate((trg_ds.train[1], trg_ds.dev[1]), axis=0)
            test_x = trg_ds.test[0].reshape((-1, MAX_LEN * vec_dim, 1))
            test_y = trg_ds.test[1]

            class_weight = compute_class_weight('balanced', np.unique(train_y), train_y)
            # print('class weights: {}'.format(class_weight))
            weights = np.zeros(train_x.shape[0], dtype=np.float32)
            for t, w in enumerate(class_weight):
                weights[train_y == t] = w

            test_scores = []
            best_dev = 0
            test_f1 = None
            test_pred = None
            for C in args.C:
                tf.reset_default_graph()
                with tf.Session(config=config) as sess:
                    cnn = SentiCNN(sess, vec_dim, (2 if is_binary else 4),
                                   args.learning_rate, args.batch_size, args.epochs, args.filters, args.dropout, C, args.clip)
                    cnn.initialize()
                    cnn.fit(train_x, train_y, dev_x, dev_y, weights)
                    pred = cnn.predict(test_x)
                    test_scores.append(f1_score(test_y, pred, average='macro'))
                    if cnn.best_score_ > best_dev:
                        best_dev = cnn.best_score_
                        test_f1 = test_scores[-1]
                        test_pred = pred

            print('------------------------------------------------------')
            print('Is binary: {}'.format(is_binary))
            print('Result for {}:'.format(infile))
            print('Test f1 scores: {}'.format(test_scores))
            print('Test f1 macro: {:.4f}'.format(test_f1))
            print('Best dev score: {:.4f}'.format(best_dev))
            print('Confusion matrix:')
            print(confusion_matrix(test_y, test_pred))
            print('------------------------------------------------------')

            if args.output is not None:
                with open(args.output, 'a', encoding='utf-8') as fout:
                    fout.write(('{},{},{},{},{},' + '{:.4f},' * len(args.C) + '{:.4f},{:.4f}\n').format(infile, src_lang,
                                                                                                        trg_lang, model,
                                                                                                        is_binary, *test_scores,
                                                                                                        test_f1, best_dev))
    for f in glob.glob(TMP_FILE + '*'):
        os.remove(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('W',
                        nargs='+',
                        help='checkpoint files')
    parser.add_argument('--setting',
                        choices=['binary', '4-class', 'both'],
                        default='both',
                        help='classification setting')
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
    parser.add_argument('--filters',
                        help='number of conv filters (default: 64)',
                        default=64,
                        type=int)
    parser.add_argument('--dropout',
                        help='dropout rate (default: 0.5)',
                        default=0.5,
                        type=float)
    parser.add_argument('-C', '--C',
                        nargs='+',
                        type=float,
                        default=[0.4, 0.8, 1.6, 3],
                        help='regularization parameter (default: 0.03)')
    parser.add_argument('--clip',
                        action='store_true',)
    parser.add_argument('-o', '--output',
                        help='output file')
    parser.add_argument('--debug',
                        help='print debug info',
                        action='store_const',
                        dest='loglevel',
                        default=logging.INFO,
                        const=logging.DEBUG)

    parser.set_defaults(clip=True)

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel,
                        format='%(asctime)s: %(levelname)s: %(message)s')
    main(args)
