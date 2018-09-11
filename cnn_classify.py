import tensorflow as tf
import numpy as np
import argparse
from sklearn.metrics import f1_score
from utils import utils
import logging


class SentiCNN(object):
    """
    CNN for sentiment classification.
    """
    def __init__(self, sess, vec_dim, nclasses, learning_rate, batch_size, num_epoch):
        self.sess = sess
        self.vec_dim = vec_dim
        self.nclasses = nclasses
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self._build_graph()

    def _build_graph(self):
        self.inputs = tf.placeholder(tf.float32, shape=(None, None, 1))
        self.labels = tf.placeholder(tf.float32, shape=(None,))

        W1 = tf.get_variable('W1', (1, self.vec_dim, 32), tf.float32, initializer=tf.random_uniform_initializer(-1., 1.))
        conv1 = tf.nn.conv1d(inputs, W1, self.vec_dim, 'VALID') # shape (batch_size, length, nchannels)
        pool1 = tf.reduce_max(conv1, axis=1) # shape（batch_size, nchannels)
        self.maxpos = tf.argmax(conv1, axis=1)

        W2 = tf.get_variable('W2', (32, nclasses), tf.float32, initializer=tf.random_uniform_initializer(-1., 1.))
        b2 = tf.get_variable('b2', (nclasses,), tf.float32, initializer=tf.zeros_initializer())
        logits = tf.matmul(pool1, W2) + b2
        self.pred = tf.argmax(logits, axis=1)
        self.loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.labels, self.nclasses), logits)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    def fit(self, train_x, train_y):
        nsample = len(train_x)
        for epoch in range(self.num_epoch):
            loss = 0.
            pred = np.zeros(nsample)
            for index, offset in enumerate(range(0, nsample, self.batch_size)):
                xs = train_x[offset:offset + self.batch_size]
                ys = train_y[offset:offset + self.batch_size]
                _, loss_, pred_, maxpos_= self.sess.run([self.optimizer, self.loss, self.pred, self.maxpos], {self.inputs: xs, self.labels: ys})
                loss += loss_ * len(xs)
                pred[offset:offset + self.batch_size] = pred_
            loss /= nsample
            fscore = f1_score(train_y, pred, average='macro')
            logging.info('epoch: %d  f1_macro: %.4f  loss: %.6f' % (epoch, f1_score, loss))

    def predict(self, test_x):
        pred = self.sess.run(self.pred, {self.inputs: test_x})

    def score(self, test_x, test_y, scorer='f1_macro'):
        if scorer == 'f1_macro':
            return f1_score(test_y, self.predict(test_x), average='macro')
        else:
            raise NotImplementedError()


def make_data(X, y, embedding, vec_dim, binary):
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=64, padding='post')
    X = embedding[X].reshape((X.shape[0], vec_dim * 64))
    perm = np.random.permutation(X.shape[0])
    X, y = X[perm], y[perm]
    if binary:
        y = (y >= 2).astype(np.int32)
    return X, y


def main(args):
    logging.info(str(args))
    source_wordvec = utils.WordVecs(args.source_embedding, normalize=args.normalize)
    source_dataset = utils.SentimentDataset(args.source_dataset).to_index(source_wordvec)
    train_x, train_y = make_data(*source_dataset.train, source_wordvec.embedding, args.vector_dim, args.binary)
    test_x, test_y = make_data(*source_dataset.test, source_wordvec.embedding, args.vector_dim, args.binary)
    with tf.Session() as sess:
        model = SentiCNN(sess, args.vector_dim, (2 if args.binary else 4), args.learning_rate, args.batch_size, args.epochs)
        model.fit(train_x, train_y)
        logging.info('Test f1_macro: %.4f' % model.score(test_x, test_y))


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
                        help='training epochs (default: 50)',
                        default=50,
                        type=int)
    parser.add_argument('-bs', '--batch_size',
                        help='training batch size (default: 50)',
                        default=50,
                        type=int)
    parser.add_argument('-se', '--source_embedding',
                        help='monolingual word embedding of the source language (default: ./emb/en.bin)',
                        default='./emb/en.bin')
    parser.add_argument('-sd', '--source_dataset',
                        help='sentiment dataset of the source language',
                        default='./datasets/en/opener_sents/')
    parser.add_argument('-vd', '--vector_dim',
                        help='dimension of each word vector (default: 300)',
                        default=300,
                        type=int) 
    parser.add_argument('--normalize',
                        help='mean center and normalize word vectors',
                        action='store_true')
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