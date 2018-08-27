"""
train the bilingual word embedding using projection loss and classification loss

author: fyl
"""
import tensorflow as tf
import numpy as np
import argparse
from pprint import pprint
from sklearn.metrics import f1_score, accuracy_score
import logging
from utils import utils


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

    def __init__(self, sess, src_emb, tgt_emb, dictionary, savepath, binary=False):
        self.nclass = 2 if binary else 4
        self.source_emb_obj = src_emb
        self.target_emb_obj = tgt_emb
        self.dict_obj = dictionary
        self.sess = sess
        self.savepath = savepath
        self._build_graph()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def _build_graph(self):
        """
        Build the model.
        """

        def get_projection_loss(source_emb, target_emb, dictionary):
            """
            Given the source language embedding, target language embedding and a bilingual dictionary,
            compute the projection loss.
            """
            source_ids, target_ids = dictionary[:, 0], dictionary[:, 1]

            proj_loss = tf.reduce_sum(tf.squared_difference(tf.nn.embedding_lookup(
                source_emb, source_ids), tf.nn.embedding_lookup(target_emb, target_ids)))
            return proj_loss

        def softmax_layer(input):
            """
            compute $$ logits = input * P $$,
            where the shape of logits is (None, n_classes)
            doesn't perform softmax
            """
            with tf.variable_scope('softmax', reuse=tf.AUTO_REUSE):
                P = tf.get_variable('P', (args.vec_dim, self.nclass), dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-1., 1.))
                b = tf.get_variable(
                    'b', (self.nclass,), dtype=tf.float32, initializer=tf.constant_initializer(0.))
            return tf.matmul(input, P) + b

        def get_projected_embeddings(source_original_emb, target_original_emb):
            """
            Given the original embeddings, calculate the projected word embeddings.
            """
            with tf.variable_scope('projection', reuse=tf.AUTO_REUSE):
                W_source = tf.get_variable(
                    'W_source', dtype=tf.float32, initializer=tf.constant(np.identity(args.vec_dim, dtype=np.float32)))
                W_target = tf.get_variable(
                    'W_target', dtype=tf.float32, initializer=tf.constant(np.identity(args.vec_dim, dtype=np.float32)))
            source_emb = tf.matmul(source_original_emb,
                                   W_source, name='map_source')
            target_emb = tf.matmul(target_original_emb,
                                   W_target, name='map_target')
            return source_emb, target_emb

        self.source_original_emb = tf.placeholder(
            tf.float32, shape=(None, args.vec_dim))
        self.target_original_emb = tf.placeholder(
            tf.float32, shape=(None, args.vec_dim))
        self.corpus = tf.placeholder(tf.int32, shape=(None, 256))
        self.labels = tf.placeholder(tf.int32, shape=(None,))
        self.dictionary = tf.placeholder(tf.int32, shape=(None, 2))
        self.corpus_test = tf.placeholder(tf.int32, shape=(None, 256))

        source_emb, target_emb = get_projected_embeddings(
            self.source_original_emb, self.target_original_emb)

        # compute projection loss
        self.proj_loss = get_projection_loss(
            source_emb, target_emb, self.dictionary)

        # compute classification loss
        sents = tf.reduce_mean(tf.nn.embedding_lookup(
            source_emb, self.corpus), axis=1)  # shape: (None, 300)
        hypothesis = softmax_layer(sents)
        self.classification_loss = tf.losses.softmax_cross_entropy(
            tf.one_hot(self.labels, self.nclass), hypothesis)

        # compute full loss
        self.loss = (1 - args.alpha) * self.classification_loss + \
            args.alpha * self.proj_loss

        # compute accuracy counts
        self.pred = tf.argmax(hypothesis, axis=1, output_type=tf.int32)
        self.acc = tf.reduce_mean(tf.to_float(
            tf.equal(self.pred, self.labels)))  # tensor

        # predict test labels:
        sents_test = tf.reduce_mean(tf.nn.embedding_lookup(
            target_emb, self.corpus_test), axis=1)
        hypothesis_test = softmax_layer(sents_test)
        self.pred_test = tf.argmax(
            hypothesis_test, axis=1, output_type=tf.int32)

        self.global_step = tf.Variable(0,
                                       dtype=tf.int32,
                                       trainable=False,
                                       name='global_step')

        self.optimizer = tf.train.AdamOptimizer(
            args.learning_rate).minimize(self.loss, global_step=self.global_step)

    def save(self, path):
        """
        Save the model to the given path.
        """
        self.saver.save(self.sess, path, global_step=self.global_step)

    def load(self, path):
        self.saver.restore(self.sess, path)

    def fit(self, train_x, train_y):
        """
        train the model.
        """
        nsample = len(train_x)
        nbatch = nsample // args.batch_size
        logging.debug('accuracy before training: %.2f' % self.sess.run(self.acc, feed_dict={
            self.source_original_emb: self.source_emb_obj,
            self.target_original_emb: self.target_emb_obj,
            self.dictionary: self.dict_obj,
            self.corpus: train_x[:100],
            self.labels: train_y[:100],
        }))
        for epoch in range(args.epochs):
            closs, ploss, loss, acc = 0., 0., 0., 0.
            for index, offset in enumerate(range(0, nsample, args.batch_size)):
                xs = train_x[offset:offset + args.batch_size]
                ys = train_y[offset:offset + args.batch_size]
                feed_dict = {
                    self.source_original_emb: self.source_emb_obj,
                    self.target_original_emb: self.target_emb_obj,
                    self.dictionary: self.dict_obj,
                    self.corpus: xs,
                    self.labels: ys,
                }
                closs_, ploss_, loss_, acc_, _ = self.sess.run(
                    [self.classification_loss, self.proj_loss, self.loss, self.acc, self.optimizer], feed_dict=feed_dict)

                closs += closs_
                ploss += ploss_
                loss += loss_
                acc += acc_

            closs, ploss, loss, acc = closs / nbatch, ploss / \
                nbatch, loss / nbatch, acc / nbatch
            logging.info('epoch: %d  loss: %.4f  class_loss: %.4f  proj_loss: %.4f  train_acc: %.2f' %
                         (epoch, loss, closs, ploss, acc))
            if (epoch + 1) % 10 == 0:
                self.save(self.savepath)

    def predict(self, test_x):
        feed_dict = {
            self.target_original_emb: self.target_emb_obj,
            self.corpus_test: test_x,
        }
        return self.sess.run(self.pred_test, feed_dict=feed_dict)

    def predict_source(self, test_x):
        feed_dict = {
            self.source_original_emb: self.source_emb_obj,
            self.corpus: test_x,
        }
        return self.sess.run(self.pred, feed_dict=feed_dict)

    def evaluate(self, test_x, test_y):
        """
        Compute the accuracy given the test examples (in source language).
        """
        feed_dict = {
            self.source_original_emb: self.source_emb_obj,
            self.target_original_emb: self.target_emb_obj,
            self.dictionary: self.dict_obj,
            self.corpus: test_x,
            self.labels: test_y,
        }
        acc = self.sess.run(self.acc, feed_dict=feed_dict)
        logging.info('test accuracy (source language): %.4f' % acc)


def load_data(binary=False):
    """
    Return the data in numpy arrays.
    """
    source_wordvec = utils.WordVecs(args.source_embedding)
    target_wordvec = utils.WordVecs(args.target_embedding)
    args.vec_dim = source_wordvec.vec_dim

    source_pad_id = source_wordvec.add_word('<PAD>', np.zeros(300))
    target_pad_id = target_wordvec.add_word('<PAD>', np.zeros(300))

    dict_obj = utils.BilingualDict(args.dictionary).filter(
        lambda x: x[0] != '-').get_indexed_dictionary(source_wordvec, target_wordvec)

    source_dataset = utils.SentimentDataset(
        args.source_dataset).to_index(source_wordvec)
    target_dataset = utils.SentimentDataset(
        args.target_dataset).to_index(target_wordvec)

    train_x = tf.keras.preprocessing.sequence.pad_sequences(
        source_dataset.train[0], maxlen=256, value=source_pad_id)
    train_y = source_dataset.train[1]
    perm = np.random.permutation(train_x.shape[0])
    train_x, train_y = train_x[perm], train_y[perm]

    test_x = tf.keras.preprocessing.sequence.pad_sequences(
        target_dataset.train[0], maxlen=256, value=target_pad_id)
    test_y = target_dataset.train[1]
    perm = np.random.permutation(test_x.shape[0])
    test_x, test_y = test_x[perm], test_y[perm]

    test_y = (test_y >= 2).astype(np.int32)
    train_y = (train_y >= 2).astype(np.int32)

    return source_wordvec, target_wordvec, dict_obj, train_x, train_y, test_x, test_y


def evaluate(pred, true_y, binary=False):
    print(pred[:50])
    print(true_y[:50])
    acc = accuracy_score(true_y, pred)
    if binary:
        fscore = f1_score(true_y, pred, pos_label=0)
    else:
        fscore = f1_score(true_y, pred, average='macro')
    logging.info('f1_score: %.4f    accuracy: %.2f' % (fscore, acc))


def main(args):
    logging.info('fitting BLSE model with parameters: %s' % str(args))
    source_wordvec, target_wordvec, dict_obj, train_x, train_y, test_x, test_y = load_data(binary=args.binary)  # numpy array
    with tf.Session() as sess:
        model = BLSE(sess, source_wordvec.embedding, target_wordvec.embedding,
                     dict_obj, args.save_path, binary=args.binary)

        if args.model != '':
            model.load(args.model)

        evaluate(model.predict(test_x), test_y, args.binary)

        model.fit(train_x, train_y)
        model.save(args.save_path)

        evaluate(model.predict(test_x), test_y, args.binary)

        pprint([' '.join([str(w) for w in line if w != '<PAD>'])
                for line in source_wordvec.index2word(train_x[:30])])
        print()
        print(model.predict_source(train_x[:30]))
        print()
        print(train_y[:30])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-bi', '--binary',
                        help='use 2-class set up',
                        action='store_true')
    parser.add_argument('-sl', '--source_lang',
                        help='source language: en/es/ca/eu (default: en)',
                        default='eu')
    parser.add_argument('-tl', '--target_lang',
                        help='target language: en/es/ca/eu (default: es)',
                        default='es')
    parser.add_argument('-a', '--alpha',
                        help="trade-off between projection and classification objectives (default: 0.001)",
                        default=0.001,
                        type=float)
    parser.add_argument('-lr', '--learning_rate',
                        help='learning rate (default: 0.03)',
                        type=float,
                        default=0.03)
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
                        default='./emb/en.bin')
    parser.add_argument('-te', '--target_embedding',
                        help='monolingual word embedding of the target language (default: ./emb/es.bin)',
                        default='./emb/es.bin')
    parser.add_argument('-d', '--dictionary',
                        help='bilingual dictionary of source and target language (default: ./lexicons/bingliu/en-es.txt',
                        default='./lexicons/bingliu/en-eu.txt')
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
                        default='./checkpoints/')
    parser.add_argument('--model',
                        help='restore from trained model',
                        type=str,
                        default='')
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel,
                        format='%(asctime)s: %(levelname)s: %(message)s')

    main(args)
