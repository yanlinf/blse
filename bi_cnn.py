import tensorflow as tf
import numpy as np
import argparse
from sklearn.metrics import f1_score
from utils import utils
import logging


class BiSentiCNN(object):
    """
    CNN for sentiment classification.
    """

    def __init__(self, sess, vec_dim, nclasses, src_lr, trg_lr, src_bs,
                 trg_bs, src_epochs, trg_epochs, num_filters, dropout):
        self.sess = sess
        self.vec_dim = vec_dim
        self.nclasses = nclasses
        self.src_lr = src_lr
        self.trg_lr = trg_lr
        self.src_bs = src_bs
        self.trg_bs = trg_bs
        self.src_epochs = src_epochs
        self.trg_epochs = trg_epochs
        self.num_filters = num_filters
        self.dropout = dropout
        self._build_graph()
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        def conv_pool_dropout(inputs, keep_prob):
            with tf.variable_scope('conv', reuse=tf.AUTO_REUSE):
                W1 = tf.get_variable('W1', (self.vec_dim, 1, self.num_filters), tf.float32, initializer=tf.random_uniform_initializer(-1., 1.))
                conv = tf.nn.conv1d(inputs, W1, self.vec_dim, 'VALID')  # shape (batch_size, length, nchannels)
                relu = tf.nn.relu(conv)
                pool = tf.reduce_max(relu, axis=1)
                dropout = tf.nn.dropout(pool, keep_prob)
                return dropout  # shape (batch_size, nchannels)

        def softmax(inputs):
            with tf.variable_scope('softmax', reuse=tf.AUTO_REUSE):
                W2 = tf.get_variable('W2', (self.num_filters, self.nclasses), tf.float32, initializer=tf.random_uniform_initializer(-1., 1.))
                b2 = tf.get_variable('b2', (self.nclasses,), tf.float32, initializer=tf.zeros_initializer())
            return inputs @ W2 + b2

        def project(inputs):
            with tf.variable_scope('project', reuse=tf.AUTO_REUSE):
                U = tf.get_variable('U', (self.vec_dim, self.vec_dim), tf.float32, initializer=tf.constant_initializer(np.identity(self.vec_dim)))
                self.U = U
                return tf.reshape((tf.reshape(inputs, (-1, self.vec_dim)) @ U), (-1, 64 * self.vec_dim))

        self.keep_prob = tf.placeholder(tf.float32)
        self.src_x = tf.placeholder(tf.float32, shape=(None, None, 1))
        self.src_y = tf.placeholder(tf.int32, shape=(None,))
        self.trg_x = tf.placeholder(tf.float32, shape=(None, None, 1))
        self.trg_y = tf.placeholder(tf.int32, shape=(None,))

        src_logits = softmax(conv_pool_dropout(self.src_x, self.keep_prob))
        trg_logits = softmax(conv_pool_dropout(project(self.trg_x, self.keep_prob)))
        src_loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.src_y, self.nclasses), src_logits)
        trg_loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.trg_y, self.nclasses), trg_logits)
        src_pred = tf.argmax(src_logits, axis=1)
        trg_pred = tf.argmax(trg_logits, axis=1)
        src_op = tf.train.AdamOptimizer(self.src_lr).minimize(self.src_loss)
        trg_op = tf.train.AdamOptimizer(self.trg_lr).minimize(self.trg_loss, var_list=self.U)

        self.src_loss = src_loss
        self.trg_loss = trg_loss
        self.src_pred = src_pred
        self.trg_pred = trg_pred
        self.src_op = src_op
        self.trg_op = trg_op

    def fit(self, src_x, src_y, trg_x, trg_y, val_x=None, val_y=None):
        # train W1, W2, b using src_x and src_y
        nsample = len(src_x)
        for epoch in range(self.src_epochs):
            src_loss = 0.
            src_pred = np.zeros(nsample)
            for i in range(0, nsample, self.batch_size):
                j = min(nsample, i + self.batch_size)
                feed_dict = {
                    self.src_x: src_x[i:j],
                    self.src_y: src_y[i:j],
                    self.keep_prob: self.dropout,
                }
                _, src_loss_, src_pred_ = self.sess.run([self.src_op, self.src_loss, self.src_pred], feed_dict)
                src_loss += loss_ * (j - i)
                src_pred[i:j] = pred_
            src_loss /= nsample
            fscore = f1_score(src_y, src_pred, average='macro')
            logging.info('epoch: %d  f1_macro: %.4f  loss: %.6f' % (epoch, fscore, src_loss))

            if val_x is not None and val_y is not None:
                logging.info('Test f1_macro: %.4f' % self.score(val_x, val_y))

        # train U using trg_x and trg_y
        nsample = len(trg_x)
        for epoch in range(self.trg_epochs):
            trg_loss = 0.
            trg_pred = np.zeros(nsample)
            for i in range(0, nsample, self.batch_size):
                j = min(nsample, i + self.batch_size)
                feed_dict = {
                    self.trg_x: trg_x[i:j],
                    self.trg_y: trg_y[i:j],
                    self.keep_prob: 1.,
                }
                _, trg_loss_, trg_pred_ = self.sess.run([self.trg_op, self.trg_loss, self.trg_pred], feed_dict)
                trg_loss += loss_ * (j - i)
                trg_pred[i:j] = pred_
            trg_loss /= nsample
            fscore = f1_score(trg_y, trg_pred, average='macro')
            logging.info('epoch: %d  f1_macro: %.4f  loss: %.6f' % (epoch, fscore, trg_loss))

    def predict(self, test_x):
        pred = self.sess.run(self.src_pred, {self.src_x: test_x, self.keep_prob: 1.})
        return pred

    def score(self, test_x, test_y, scorer='f1_macro'):
        if scorer == 'f1_macro':
            return f1_score(test_y, self.predict(test_x), average='macro')
        else:
            raise NotImplementedError()

    def save(self, savepath):
        tf.train.Saver().save(self.sess, savepath)

    @property
    def W_target(self):
        return self.sess.run(self.U)



def make_data(X, y, embedding, vec_dim, binary, pad_id, shuffle=True):
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=64, padding='post', value=pad_id)
    X = embedding[X].reshape((X.shape[0], vec_dim * 64, 1))
    if shuffle:
        perm = np.random.permutation(X.shape[0])
        X, y = X[perm], y[perm]
    if binary:
        y = (y >= 2).astype(np.int32)
    return X, y


def main(args):
    logging.info(str(args))

    src_wv = utils.WordVecs(args.source_embedding, normalize=args.normalize)
    trg_wv = utils.WordVecs(args.source_embedding, normalize=args.normalize)
    src_pad_id = src_wv.add_word('<PAD>', np.zeros(300))
    trg_pad_id = trg_wv.add_word('<PAD>', np.zeros(300))
    src_dataset = utils.SentimentDataset(args.source_dataset).to_index(src_wv)
    trg_dataset = utils.SentimentDataset(args.target_dataset).to_index(trg_wv)
    src_x, src_y = make_data(*src_dataset.train, src_wv.embedding, args.vector_dim, args.binary, src_pad_id)
    test_x, test_y = make_data(*src_dataset.test, src_wv.embedding, args.vector_dim, args.binary, src_pad_id)
    trg_x, trg_y = make_data(*trg_dataset.train, trg_wv.embedding, args.vector_dim, args.binary, trg_pad_id)

    with tf.Session() as sess:
        model = BiSentiCNN(sess=sess, vec_dim=args.vector_dim, nclasses=(2 if args.binary else 4),
                           src_lr=args.source_learning_rate, trg_lr=args.target_learning_rate,
                           src_bs=args.source_batch_size, trg_bs=args.target_batch_size,
                           src_epochs=args.source_epochs, trg_epochs=args.target_epochs,
                           num_filters=args.filters, dropout=args.dropout)
        model.fit(src_x, src_y, trg_x, trg_y, test_x, test_y)
        model.save(args.save_path)

        src_emb = src_wv.embedding
        trg_emb = np.dot(trg_wv.embedding, model.W_target)
        trg_indices = np.argmax(np.dot(src_emb[gold_dict[:, 0]], trg_emb.T), axis=1)
        acc = np.mean((trg_indices == gold_dict[:, 1]).astype(np.int32))
        logging.info('Accuracy on bilingual dictionary induction: %.4f' % acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-bi', '--binary', action='store_true', help='use 2-class set up')
    parser.add_argument('-slr', '--source_learning_rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('-tlr', '--target_learning_rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('-sep', '--source_epochs', default=200, type=int, help='training epochs (default: 200)')
    parser.add_argument('-tep', '--target_epochs', default=200, type=int, help='training epochs (default: 200)')
    parser.add_argument('-sbs', '--source_batch_size', default=50, type=int, help='training batch size (default: 50)')
    parser.add_argument('-tbs', '--target_batch_size', default=50, type=int, help='training batch size (default: 50)')
    parser.add_argument('-f', '--filters', default=64, type=int, help='number of conv filters (default: 64)')
    parser.add_argument('-dp', '--dropout', default=0.5, type=float, help='dropout rate (default: 0.5)')
    parser.add_argument('-se', '--source_embedding', default='./emb/en.bin', help='source embedding (default: ./emb/en.bin)')
    parser.add_argument('-te', '--target_embedding', default='./emb/es.bin', help='target embedding (default: ./emb/es.bin)')
    parser.add_argument('-sd', '--source_dataset', default='./datasets/en/opener_sents/', help='sentiment dataset of the source language')
    parser.add_argument('-st', '--target_dataset', default='./datasets/es/opener_sents/', help='sentiment dataset of the target language')
    parser.add_argument('-vd', '--vector_dim', default=300, type=int, help='dimension of each word vector (default: 300)')
    parser.add_argument('-gd', '--gold_dictionary', default='./lexicons/apertium/en-es.txt', help='gold bilingual dictionary for evaluation')
    parser.add_argument('--normalize', action='store_true', help='mean center and normalize word vectors')
    parser.add_argument('--save_path', type=str, help='file to save the trained parameters')
    parser.add_argument('--debug', action='store_const', dest='loglevel', default=logging.INFO, const=logging.DEBUG, help='print debug info')

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel,
                        format='%(asctime)s: %(levelname)s: %(message)s')
    main(args)
