"""
train the bilingual word embedding using projection loss and classification loss

author: fyl
"""
import tensorflow as tf
import numpy as np
import argparse
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
    """

    def __init__(self, sess, src_emb, tgt_emb, dictionary, savepath):
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
                P = tf.get_variable('P', (args.vec_dim, 4), dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-1., 1.))
                b = tf.get_variable('b', (4,), dtype=tf.float32)
            return tf.matmul(input, P) + b

        def get_projected_embeddings(source_original_emb, target_original_emb):
            """
            Given the original embeddings, calculate the projected word embeddings.
            """
            with tf.variable_scope('projection', reuse=tf.AUTO_REUSE):
                W_source = tf.get_variable(
                    'W_source', (args.vec_dim, args.vec_dim), dtype=tf.float32, initializer=tf.random_uniform_initializer(-1., 1.))
                W_target = tf.get_variable(
                    'W_target', (args.vec_dim, args.vec_dim), dtype=tf.float32, initializer=tf.random_uniform_initializer(-1., 1.))
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

        source_emb, target_emb = get_projected_embeddings(
            source_original_emb, target_original_emb)

        # compute projection loss
        self.proj_loss = get_projection_loss(
            source_emb, target_emb, self.dictionary)

        # compute classification loss
        sents = tf.reduce_mean(tf.nn.embedding_lookup(
            source_emb, self.corpus), axis=1)  # shape: (None, 300)
        hypothesis = softmax_layer(sents)
        self.classification_loss = tf.losses.softmax_cross_entropy(
            tf.one_hot(y, 4), hypothesis)

        # compute full loss
        self.loss = args.alpha * self.classification_loss + \
            (1 - args.alpha) * self.proj_loss

        # compute accuracy counts
        self.pred = tf.argmax(hypothesis, axis=1, output_type=tf.int32)
        self.acc = tf.reduce_mean(tf.to_float(
            tf.equal(pred, labels)))  # tensor

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

    def fit(self, train_x, train_y):
        """
        train the model.
        """
        nsample = len(train_x)
        nbatch = nsample // args.batch_size
        for epoch in range(args.epochs):
            closs, ploss, loss, acc = 0., 0.
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
                closs_, ploss_, loss_, acc_, _, self.sess.run(
                    [self.classification_loss, self.proj_loss, self.loss, self.acc, self.optimizer], feed_dict=feed_dict)

                closs += closs_
                ploss += ploss_
                loss += loss_
                acc += acc_

            closs, ploss, loss, acc = closs / nbatch, ploss / \
                nbatch, loss / nbatch, acc / nbatch
            print('epoch: %d  loss: %.4f  class_loss: %.4f  proj_loss: %.4f  train_acc: %.2f=' %
                  (epoch, loss, closs, ploss, acc))
            if (epoch + 1) % 10 == 0:
                self.save(self.savepath)

    def evaluate(self, test_x, test_y):
        """
        Compute the accuracy given the test examples.
        """
        feed_dict = {
            self.source_original_emb: self.source_emb_obj,
            self.target_original_emb: self.target_emb_obj,
            self.dictionary: self.dict_obj,
            self.corpus: test_x,
            self.labels: test_y,
        }
        acc = self.sess.run(self.acc, feed_dict=feed_dict)
        print('test accuracy: %.4f' % acc)


def load_data():
    """
    Return the data in numpy arrays.
    """
    source_wordvec = utils.WordVecs(args.source_embedding)
    target_wordvec = utils.WordVecs(args.target_embedding)
    args.vec_dim = source_wordvec.vec_dim

    pad_id = source_wordvec.add_word('<PAD>', np.zeros(300))

    dict_obj = utils.BilingualDict(args.dictionary).filter(
        lambda x: x[0] != '-').get_indexed_dictionary(source_wordvec, target_wordvec)

    senti_dataset = utils.SentimentDataset(
        args.sentiment_dataset).to_index(source_wordvec)
    train_x = tf.keras.preprocessing.sequence.pad_sequences(
        senti_dataset.train[0], maxlen=256, value=pad_id)
    test_x = tf.keras.preprocessing.sequence.pad_sequences(
        senti_dataset.test[0], maxlen=256, value=pad_id)
    train_y = senti_dataset.train[1]
    test_y = senti_dataset.test[1]

    return source_wordvec.embedding, target_wordvec.embedding, dict_obj, train_x, train_y, test_x, test_y


def main():
    source_emb_obj, target_emb_obj, dict_obj, train_x, train_y, test_x, test_y = load_data()  # numpy array

    with tf.Session() as sess:
        model = BLSE(sess, source_emb_obj, target_emb_obj, dict_obj)
        model.fit(train_x, train_y)
        model.evaluate(test_x, test_y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument('-sd', '--sentiment_dataset',
                        help='sentiment dataset of the source language',
                        default='./datasets/en/opener_sents/')
    parser.add_argument('-vd', '--vector_dim',
                        help='dimension of each word vector (default: 300)',
                        default=300,
                        type=int)
    parser.add_argument('--debug',
                        help='print debug info',
                        action='store_true')
    parser.add_argument('--save_path',
                        help='the dictionary to store the trained model',
                        default='./checkpoints/')
    args = parser.parse_args()
    main()
