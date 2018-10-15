import tensorflow as tf
import numpy as np
import argparse
from sklearn.metrics import f1_score
from utils.dataset import *
from utils.math import *
from utils.bdi import *
import logging


def DEBUG(s):
    logging.debug('[DEBUG] ' + str(s))


class BiSentiLR(object):

    def __init__(self, sess, vec_dim, nclasses, src_lr, trg_lr, src_bs,
                 trg_bs, src_epochs, trg_epochs, orthogonal, summaries_dir,
                 W_trg):
        self.sess = sess
        self.vec_dim = vec_dim
        self.nclasses = nclasses
        self.src_lr = src_lr
        self.trg_lr = trg_lr
        self.src_bs = src_bs
        self.trg_bs = trg_bs
        self.src_epochs = src_epochs
        self.trg_epochs = trg_epochs
        self.orthogonal = orthogonal
        self.summaries_dir = summaries_dir
        self.W_trg = W_trg
        self._build_graph()
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        def softmax(inputs):
            with tf.variable_scope('softmax', reuse=tf.AUTO_REUSE):
                W2 = tf.get_variable('W2', (self.vec_dim, self.nclasses), tf.float32, initializer=tf.random_uniform_initializer(-1., 1.))
                b2 = tf.get_variable('b2', (self.nclasses,), tf.float32, initializer=tf.zeros_initializer())
                common_summ.append(tf.summary.histogram('W2', W2))
                common_summ.append(tf.summary.histogram('b2', b2))
            return inputs @ W2 + b2

        def project(inputs):
            with tf.variable_scope('project', reuse=tf.AUTO_REUSE):
                U = tf.get_variable('U', (self.vec_dim, self.vec_dim), tf.float32, initializer=(tf.random_uniform_initializer(-1., 1.) if self.W_trg is None else tf.constant_initializer(self.W_trg)))
                self.U = U
                trg_summ.append(tf.summary.histogram('U', U))
                return inputs @ U

        self.src_x = tf.placeholder(tf.float32, shape=(None, self.vec_dim))
        self.src_y = tf.placeholder(tf.int32, shape=(None,))
        self.trg_x = tf.placeholder(tf.float32, shape=(None, self.vec_dim))
        self.trg_y = tf.placeholder(tf.int32, shape=(None,))
        self.X_src = tf.placeholder(tf.float32, shape=(None, self.vec_dim))
        self.X_trg = tf.placeholder(tf.float32, shape=(None, self.vec_dim))

        src_summ = []
        trg_summ = []
        common_summ = []

        src_logits = softmax(self.src_x)
        trg_logits = softmax(project(self.trg_x))
        src_loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.src_y, self.nclasses), src_logits)
        trg_loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.trg_y, self.nclasses), trg_logits)
        src_pred = tf.argmax(src_logits, axis=1)
        trg_pred = tf.argmax(trg_logits, axis=1)
        src_op = tf.train.AdamOptimizer(self.src_lr).minimize(src_loss)
        trg_op = tf.train.GradientDescentOptimizer(self.trg_lr).minimize(trg_loss, var_list=self.U)

        proj_loss = tf.reduce_sum(tf.squared_difference(self.X_src, project(self.X_trg)))

        src_summ.append(tf.summary.scalar('src_loss', src_loss))
        trg_summ.append(tf.summary.scalar('trg_loss', trg_loss))

        self.src_merged = tf.summary.merge(src_summ + common_summ)
        self.trg_merged = tf.summary.merge(trg_summ + common_summ)

        sx, ux, vx = tf.svd(self.U)
        U_ortho = self.U.assign(tf.matmul(ux, vx, adjoint_b=True))

        self.proj_loss = proj_loss
        self.src_loss = src_loss
        self.trg_loss = trg_loss
        self.src_pred = src_pred
        self.trg_pred = trg_pred
        self.src_op = src_op
        self.trg_op = trg_op
        self.U_ortho = U_ortho

    def fit(self, src_x, src_y, trg_x, trg_y, src_val_x=None, src_val_y=None, trg_val_x=None, trg_val_y=None, X_src=None, X_trg=None, W_true=None):
        src_writer = tf.summary.FileWriter(self.summaries_dir + '/src', self.sess.graph)
        trg_writer = tf.summary.FileWriter(self.summaries_dir + '/trg', self.sess.graph)

        # train W1, W2, b using src_x and src_y
        nsample = len(src_x)
        cnt = 0
        for epoch in range(self.src_epochs):
            src_loss = 0.
            src_pred = np.zeros(nsample)
            for i in range(0, nsample, self.src_bs):
                j = min(nsample, i + self.src_bs)
                feed_dict = {
                    self.src_x: src_x[i:j],
                    self.src_y: src_y[i:j],
                }
                _, src_loss_, src_pred_, src_merged_ = self.sess.run([self.src_op, self.src_loss, self.src_pred, self.src_merged], feed_dict)
                src_loss += src_loss_ * (j - i)
                src_pred[i:j] = src_pred_

                src_writer.add_summary(src_merged_, cnt)
                cnt += 1

            src_loss /= nsample
            fscore = f1_score(src_y, src_pred, average='macro')
            logging.info('epoch: %d  f1_macro: %.4f  loss: %.6f' % (epoch, fscore, src_loss))

            if src_val_x is not None and src_val_y is not None:
                logging.info('Test f1_macro: %.4f' % self.score(src_val_x, src_val_y))

        logging.info('==========================================================')
        logging.info('                  Start training W_target                 ')
        logging.info('==========================================================')

        # train U using trg_x and trg_y
        nsample = len(trg_x)
        cnt = 0
        for epoch in range(self.trg_epochs):
            trg_loss = 0.
            trg_pred = np.zeros(nsample)
            for i in range(0, nsample, self.trg_bs):
                j = min(nsample, i + self.trg_bs)
                feed_dict = {
                    self.trg_x: trg_x[i:j],
                    self.trg_y: trg_y[i:j],
                }
                _, trg_loss_, trg_pred_, U_, trg_merged_ = self.sess.run([self.trg_op, self.trg_loss, self.trg_pred, self.U, self.trg_merged], feed_dict)
                trg_loss += trg_loss_ * (j - i)
                trg_pred[i:j] = trg_pred_

                trg_writer.add_summary(trg_merged_, cnt)
                cnt += 1
#                 DEBUG('trg_loss_: %.8f' % trg_loss_)
#                 DEBUG(U_[:2, :2])

            if self.orthogonal:
                _, U_ = self.sess.run([self.U_ortho, self.U])
#                 DEBUG(U_[:2, :2])

            trg_loss /= nsample
            fscore = f1_score(trg_y, trg_pred, average='macro')
            logging.info('epoch: %d  f1_macro: %.4f  loss: %.6f' % (epoch, fscore, trg_loss))

            if trg_val_x is not None and trg_val_y is not None:
                logging.info('Test f1_macro: %.4f' % self.trg_score(trg_val_x, trg_val_y))

            if X_src is not None and X_trg is not None:
                proj_loss_ = self.sess.run(self.proj_loss, {self.X_src: X_src, self.X_trg: X_trg})
                logging.info('Projection error: %.4f' % proj_loss_)
                
            if W_true is not None:
                fnorm = np.sum((self.W_target - W_true)**2)
                logging.info('F-norm error: %.4f' % fnorm)

        logging.info('==========================================================')
        DEBUG('Test f1_macro: %.4f' % self.score(src_val_x, src_val_y))

    def predict(self, test_x):
        pred = self.sess.run(self.src_pred, {self.src_x: test_x})
        return pred

    def trg_predict(self, trg_test_x):
        trg_pred = self.sess.run(self.trg_pred, {self.trg_x: trg_test_x})
        return trg_pred

    def score(self, test_x, test_y, scorer='f1_macro'):
        if scorer == 'f1_macro':
            return f1_score(test_y, self.predict(test_x), average='macro')
        else:
            raise NotImplementedError()

    def trg_score(self, trg_test_x, trg_test_y, scorer='f1_macro'):
        if scorer == 'f1_macro':
            return f1_score(trg_test_y, self.trg_predict(trg_test_x), average='macro')
        else:
            raise NotImplementedError()

    def save(self, savepath):
        tf.train.Saver().save(self.sess, savepath)

    @property
    def W_target(self):
        return self.sess.run(self.U)


def make_data(X, y, embedding, vec_dim, binary, pad_id, shuffle=True):
    X_new = np.zeros((len(X), vec_dim), dtype=np.float32)
    for i, row in enumerate(X):
        if len(row) > 0:
            X_new[i] = np.mean(embedding[row], axis=0)
        else:
            logging.warning('ZERO LENGTH EXAMPLE')
    X = X_new
    if shuffle:
        perm = np.random.permutation(X.shape[0])
        X, y = X[perm], y[perm]
    if binary:
        y = (y >= 2).astype(np.int32)
    return X, y


def main(args):
    logging.info(str(args))

    src_wv = WordVecs(args.source_embedding, normalize=args.normalize)
    trg_wv = WordVecs(args.target_embedding, normalize=args.normalize)
    src_pad_id = src_wv.add_word('<PAD>', np.zeros(300))
    trg_pad_id = trg_wv.add_word('<PAD>', np.zeros(300))
    src_dataset = SentimentDataset(args.source_dataset).to_index(src_wv)
    trg_dataset = SentimentDataset(args.target_dataset).to_index(trg_wv)
    src_x, src_y = make_data(*src_dataset.train, src_wv.embedding, args.vector_dim, args.binary, src_pad_id)
    src_test_x, src_test_y = make_data(*src_dataset.test, src_wv.embedding, args.vector_dim, args.binary, src_pad_id)
    trg_x, trg_y = make_data(*trg_dataset.train, trg_wv.embedding, args.vector_dim, args.binary, trg_pad_id)
    trg_test_x, trg_test_y = make_data(*trg_dataset.test, trg_wv.embedding, args.vector_dim, args.binary, trg_pad_id)
    gold_dict = BilingualDict(args.gold_dictionary).get_indexed_dictionary(src_wv, trg_wv)
    X_src = src_wv.embedding[gold_dict[:, 0]]
    X_trg = trg_wv.embedding[gold_dict[:, 1]]

    u, s, vt = np.linalg.svd(np.dot(X_src.T, X_trg))
    W_true = np.dot(vt.T, u.T)

    with tf.Session() as sess:
        model = BiSentiLR(sess=sess, vec_dim=args.vector_dim, nclasses=(2 if args.binary else 4),
                          src_lr=args.source_learning_rate, trg_lr=args.target_learning_rate,
                          src_bs=args.source_batch_size, trg_bs=args.target_batch_size,
                          src_epochs=args.source_epochs, trg_epochs=args.target_epochs,
                          orthogonal=args.orthogonal, summaries_dir=args.summaries_dir, W_trg=None)
        model.fit(src_x, src_y, trg_x, trg_y, src_test_x, src_test_y, trg_test_x, trg_test_y, X_src, X_trg, W_true)
        model.save(args.save_path)

        u, s, vt = np.linalg.svd(model.W_target)
        W_trg = np.dot(u, vt)
        bdi = BDI(src_wv.embedding, trg_wv.embedding, 50, args.cuda)
        trg_indices = bdi.project(W_trg).get_target_indices(gold_dict[:, 0])
        acc = np.mean((trg_indices == gold_dict[:, 1]).astype(np.int32))
        logging.info('Accuracy on bilingual dictionary induction: %.8f' % acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-bi', '--binary', action='store_true', help='use 2-class set up')
    parser.add_argument('-slr', '--source_learning_rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('-tlr', '--target_learning_rate', type=float, default=0.1, help='learning rate (default: 0.1)')
    parser.add_argument('-sep', '--source_epochs', default=200, type=int, help='training epochs (default: 200)')
    parser.add_argument('-tep', '--target_epochs', default=200, type=int, help='training epochs (default: 200)')
    parser.add_argument('-sbs', '--source_batch_size', default=50, type=int, help='training batch size (default: 50)')
    parser.add_argument('-tbs', '--target_batch_size', default=500, type=int, help='training batch size (default: 500)')
    parser.add_argument('-se', '--source_embedding', default='./emb/en.bin', help='source embedding (default: ./emb/en.bin)')
    parser.add_argument('-te', '--target_embedding', default='./emb/es.bin', help='target embedding (default: ./emb/es.bin)')
    parser.add_argument('-sd', '--source_dataset', default='./datasets/en/opener_sents/', help='sentiment dataset of the source language')
    parser.add_argument('-st', '--target_dataset', default='./datasets/es/opener_sents/', help='sentiment dataset of the target language')
    parser.add_argument('-vd', '--vector_dim', default=300, type=int, help='dimension of each word vector (default: 300)')
    parser.add_argument('-gd', '--gold_dictionary', default='./lexicons/apertium/en-es.txt', help='gold bilingual dictionary for evaluation')
    parser.add_argument('--normalize', action='store_true', help='mean center and normalize word vectors')
    parser.add_argument('--orthogonal', action='store_true', help='restrict W_target to be orthogonal during training')
    parser.add_argument('--cuda', action='store_true', help='use cuda for BDI')
    parser.add_argument('--save_path', type=str, default='./checkpoints/bilr.ckpt', help='file to save the trained parameters')
    parser.add_argument('--summaries_dir', type=str, default='./log', help='dir to save summaries')
    parser.add_argument('--debug', action='store_const', dest='loglevel', default=logging.INFO, const=logging.DEBUG, help='print debug info')

    parser.set_defaults(normalize=True, orthogonal=True, binary=True, cuda=True)

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel,
                        format='%(asctime)s: %(message)s')
    main(args)
