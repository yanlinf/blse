"""
train the bilingual word embedding using projection loss and classification loss

author: fyl
"""
import tensorflow as tf
import numpy as np
import argparse
from pprint import pprint
from sklearn.metrics import f1_score
import logging
from utils.dataset import *
from utils.math import *
from utils.bdi import *
from utils.model import *


class BLSE(object):
    """
    Bilingual sentiment embeddings.

    Parameters
    ----------
    sess: tf.Session() object
    saver: tf.train.Saver() object
    src_emb: numpy.ndarray of shape (source_emb_size, vec_dim)
    tgt_emb: numpy.ndarray of shape (source_emb_size, vec_dim)
    dictionary: numpy.ndarray of shape (dictionary_size, 2)
    binary: bool, optional (deafult: False)
    """

    def __init__(self, sess, savepath, vec_dim, alpha, learning_rate, batch_size, epochs, binary):
        self.nclass = 2 if binary else 4
        self.sess = sess
        self.savepath = savepath
        self.vec_dim = vec_dim
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self._build_graph()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def _build_graph(self):
        """
        Build the model.
        """
        def project_source(vecs):
            with tf.variable_scope('projection', reuse=tf.AUTO_REUSE):
                W_source = tf.get_variable(
                    'W_source', (self.vec_dim, self.vec_dim), dtype=tf.float32, initializer=tf.constant_initializer(np.identity(self.vec_dim)))
                self.W_source = W_source

            return tf.matmul(vecs, W_source)

        def project_target(vecs):
            with tf.variable_scope('projection', reuse=tf.AUTO_REUSE):
                W_target = tf.get_variable(
                    'W_target', (self.vec_dim, self.vec_dim), dtype=tf.float32, initializer=tf.constant_initializer(np.identity(self.vec_dim)))
                self.W_target = W_target

            return tf.matmul(vecs, W_target)

        def get_projection_loss(source_words, target_words):
            proj_loss = tf.reduce_sum(tf.squared_difference(project_source(source_words), project_target(target_words)))
            return proj_loss

        def softmax_layer(input):
            """
            compute $$ logits = input * P $$,
            where the shape of logits is (None, n_classes)
            doesn't perform softmax
            """
            with tf.variable_scope('softmax', reuse=tf.AUTO_REUSE):
                P = tf.get_variable('P', (self.vec_dim, self.nclass), dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-1., 1.))
                b = tf.get_variable('b', (self.nclass,), dtype=tf.float32, initializer=tf.zeros_initializer())
                self.P = P
                self.b = b
            return tf.matmul(input, P) + b

        self.train_x = tf.placeholder(tf.float32, shape=(None, self.vec_dim))
        self.train_y = tf.placeholder(tf.int32, shape=(None,))
        self.test_x = tf.placeholder(tf.float32, shape=(None, self.vec_dim))
        self.source_words = tf.placeholder(tf.float32, shape=(None, self.vec_dim))
        self.target_words = tf.placeholder(tf.float32, shape=(None, self.vec_dim))

        # compute projection loss
        self.proj_loss = get_projection_loss(self.source_words, self.target_words)

        # compute classification loss
        train_logits = softmax_layer(project_source(self.train_x))
        self.classification_loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.train_y, self.nclass), train_logits)

        # compute full loss
        self.loss = (1 - self.alpha) * self.classification_loss + self.alpha * self.proj_loss

        # compute accuracy counts
        self.pred_train = tf.argmax(train_logits, axis=1, output_type=tf.int32)

        # predict test labels:
        test_logits = softmax_layer(project_target(self.test_x))
        self.pred_test = tf.argmax(test_logits, axis=1, output_type=tf.int32)

        self.global_step = tf.Variable(0,
                                       dtype=tf.int32,
                                       trainable=False,
                                       name='global_step')

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

    def save(self, path):
        """
        Save the model to the given path.
        """
        self.saver.save(self.sess, path, global_step=self.global_step)

    def load(self, path):
        self.saver.restore(self.sess, path)

    def fit(self, train_x, train_y, source_words, target_words, test_x=None, test_y=None):
        """
        train the model.
        """
        nsample = len(train_x)
        nbatch = nsample // self.batch_size
        best_test_f1 = 0.

        for epoch in range(self.epochs):
            closs, ploss, loss = 0., 0., 0.
            pred = np.zeros(nsample)
            for index, offset in enumerate(range(0, nsample, self.batch_size)):
                xs = train_x[offset:offset + args.batch_size]
                ys = train_y[offset:offset + args.batch_size]
                feed_dict = {
                    self.train_x: xs,
                    self.train_y: ys,
                    self.source_words: source_words,
                    self.target_words: target_words,
                }
                closs_, ploss_, loss_, _, pred_ = self.sess.run(
                    [self.classification_loss, self.proj_loss, self.loss, self.optimizer, self.pred_train], feed_dict=feed_dict)
                pred[offset:offset + args.batch_size] = pred_
                closs += closs_
                ploss += ploss_
                loss += loss_

            closs, ploss, loss, = closs / nbatch, ploss / nbatch, loss / nbatch
            fscore = f1_score(train_y, pred, average='macro')
            logging.info('epoch: %d  loss: %.4f  class_loss: %.4f  proj_loss: %.4f  f1_macro: %.4f' % (epoch, loss, closs, ploss, fscore))
            # if (epoch + 1) % 50 == 0:
            #     self.save(self.savepath)

            if test_x is not None and test_y is not None:
                test_f1 = self.score(test_x, test_y)
                logging.info('Dev f1_macro: %.4f' % test_f1)
                if test_f1 > best_test_f1:
                    best_test_f1 = test_f1
                    best_W_src, best_W_trg, best_P, best_b = self.sess.run([self.W_source, self.W_target, self.P, self.b])
        return best_W_src, best_W_trg, best_P, best_b

    def predict(self, test_x):
        return self.sess.run(self.pred_test, feed_dict={self.test_x: test_x})

    def score(self, test_x, test_y):
        return f1_score(test_y, self.predict(test_x), average='macro')


def load_data(binary=False):
    """
    Return the data in numpy arrays.
    """
    def lookup_and_shuffle(X, y, emb, binary=False):
        X_new = np.zeros((len(X), emb.shape[1]))
        for i, line in enumerate(X):
            if len(line) == 0:
                logging.warning('ZERO LENGTH EXAMPLE')
                continue
            X_new[i] = np.mean(emb[line], axis=0)
        X = X_new

        perm = np.random.permutation(X.shape[0])
        X, y = X[perm], y[perm]
        if binary:
            y = (y >= 2).astype(np.int32)
        return X, y

    if args.pickle:
        with open(args.source_embedding, 'rb') as fin:
            source_wordvecs = pickle.load(fin)
        with open(args.target_embedding, 'rb') as fin:
            target_wordvecs = pickle.load(fin)
    else:
        source_wordvecs = WordVecs(args.source_embedding, emb_format=args.format)
        target_wordvecs = WordVecs(args.target_embedding, emb_format=args.format)

    dict_obj = BilingualDict(args.dictionary).filter(
        lambda x: x[0] != '-').get_indexed_dictionary(source_wordvecs, target_wordvecs)
    source_words = source_wordvecs.embedding[dict_obj[:, 0]]
    target_words = target_wordvecs.embedding[dict_obj[:, 1]]

    source_dataset = SentimentDataset(args.source_dataset).to_index(source_wordvecs)
    target_dataset = SentimentDataset(args.target_dataset).to_index(target_wordvecs)

    train_x, train_y = lookup_and_shuffle(*source_dataset.train, source_wordvecs.embedding, binary)
    test_x, test_y = lookup_and_shuffle(*target_dataset.test, target_wordvecs.embedding, binary)
    dev_x, dev_y = lookup_and_shuffle(*target_dataset.dev, target_wordvecs.embedding, binary)

    return source_wordvecs, target_wordvecs, source_words, target_words, train_x, train_y, test_x, test_y, dev_x, dev_y


# def evaluate(pred, true_y, binary=False):
#     acc = accuracy_score(true_y, pred)
#     if binary:
#         fscore = f1_score(true_y, pred, pos_label=0)
#     else:
#         fscore = f1_score(true_y, pred, average='macro')
#     logging.info('f1_score: %.4f    accuracy: %.2f' % (fscore, acc))


def main(args):
    logging.info(str(args))
    source_wordvecs, target_wordvecs, source_words, target_words, train_x, train_y, test_x, test_y, dev_x, dev_y = load_data(
        binary=args.binary)  # numpy array
    with tf.Session() as sess:
        model = BLSE(sess, args.save_path, args.vector_dim, args.alpha, args.learning_rate,
                     args.batch_size, args.epochs, binary=args.binary)

        if args.model is not None:
            model.load(args.model)
        else:
            W_src, W_trg, P, b = model.fit(train_x, train_y, source_words, target_words, dev_x, dev_y)

        logging.info('Test f1_macro: %.4f' % model.score(test_x, test_y))
        save_model(W_src, W_trg, args.source_lang, args.target_lang,
                   'blse', args.save_path, P=P, b=b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-bi', '--binary',
                        help='use 2-class set up',
                        action='store_true')
    parser.add_argument('-sl', '--source_lang',
                        help='source language: en/es/ca/eu (default: en)',
                        default='en')
    parser.add_argument('-tl', '--target_lang',
                        help='target language: en/es/ca/eu (default: es)',
                        default='es')
    parser.add_argument('--format',
                        choices=['word2vec_bin', 'fasttext_text'],
                        default='fasttext_text',
                        help='word embedding format')
    parser.add_argument('-a', '--alpha',
                        help="trade-off between projection and classification objectives (default: 0.001)",
                        default=0.001,
                        type=float)
    parser.add_argument('-lr', '--learning_rate',
                        help='learning rate (default: 0.01)',
                        type=float,
                        default=0.01)
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
                        default='./emb/wiki.en.vec')
    parser.add_argument('-te', '--target_embedding',
                        help='monolingual word embedding of the target language (default: ./emb/es.bin)',
                        default='./emb/wiki.es.vec')
    parser.add_argument('-d', '--dictionary',
                        help='bilingual dictionary of source and target language (default: ./lexicons/bingliu/en-es.txt',
                        default='./lexicons/bingliu/en-es.txt')
    parser.add_argument('-sd', '--source_dataset',
                        help='sentiment dataset of the source language',
                        default='./datasets/en/opener_sents/')
    parser.add_argument('-td', '--target_dataset',
                        help='sentiment dataset of the target language',
                        default='./datasets/es/opener_sents/')
    parser.add_argument('-vd', '--vector_dim',
                        help='dimension of each word vector (default: 300)',
                        default=300,
                        type=int)
    parser.add_argument('--debug',
                        help='print debug info',
                        action='store_const',
                        dest='loglevel',
                        default=logging.INFO,
                        const=logging.DEBUG)
    parser.add_argument('--save_path',
                        help='the dictionary to store the trained model',
                        default='./checkpoints/blse.bin')
    parser.add_argument('--model',
                        help='restore from trained model')
    parser.add_argument('--pickle',
                        action='store_true',
                        help='load from pickled objects')

    lang_group = parser.add_mutually_exclusive_group()
    lang_group.add_argument('--en_es', action='store_true', help='train english-spanish embedding')
    lang_group.add_argument('--en_ca', action='store_true', help='train english-catalan embedding')
    lang_group.add_argument('--en_eu', action='store_true', help='train english-basque embedding')

    args = parser.parse_args()
    if args.en_es:
        src_emb_file = 'pickle/en.bin' if args.pickle else 'emb/wiki.en.vec'
        trg_emb_file = 'pickle/es.bin' if args.pickle else 'emb/wiki.es.vec'
        parser.set_defaults(source_lang='en', target_lang='es',
                            source_embedding=src_emb_file, target_embedding=trg_emb_file, format='fasttext_text',
                            source_dataset='datasets/en/opener_sents/', target_dataset='datasets/es/opener_sents/',
                            dictionary='lexicons/bingliu/en-es.txt')
    elif args.en_ca:
        src_emb_file = 'pickle/en.bin' if args.pickle else 'emb/wiki.en.vec'
        trg_emb_file = 'pickle/ca.bin' if args.pickle else 'emb/wiki.ca.vec'
        parser.set_defaults(source_lang='en', target_lang='ca',
                            source_embedding=src_emb_file, target_embedding=trg_emb_file, format='fasttext_text',
                            source_dataset='datasets/en/opener_sents/', target_dataset='datasets/ca/opener_sents/',
                            dictionary='lexicons/bingliu/en-ca.txt')
    elif args.en_eu:
        src_emb_file = 'pickle/en.bin' if args.pickle else 'emb/wiki.en.vec'
        trg_emb_file = 'pickle/eu.bin' if args.pickle else 'emb/wiki.eu.vec'
        parser.set_defaults(source_lang='en', target_lang='eu',
                            source_embedding=src_emb_file, target_embedding=trg_emb_file, format='fasttext_text',
                            source_dataset='datasets/en/opener_sents/', target_dataset='datasets/eu/opener_sents/',
                            dictionary='lexicons/bingliu/en-eu.txt')
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel,
                        format='%(asctime)s: %(levelname)s: %(message)s')

    main(args)
