"""
train the bilingual word embedding using projection loss and classification loss

author: fyl
"""
import tensorflow as tf
import numpy as np
import argparse
from utils import utils


def get_projection_loss(source_emb, target_emb, dictionary):
    """
    Given the source language embedding, target language embedding and a bilingual dictionary,
    compute the projection loss.
    """
    source_ids, target_ids = dictionary[:, 0], dictionary[:, 1]

    proj_loss = tf.reduce_sum(tf.squared_difference(tf.nn.embedding_lookup(
        source_emb, source_ids), tf.nn.embedding_lookup(target_emb, target_ids)))
    return proj_loss


def get_classification_loss(emb, X_id, y):
    """
    Given the language embedding and the classification dataset,
    compute the classification loss.
    """
    X = []
    for sent in X_id:
        word_list = [tf.nn.embedding_lookup(emb, word_id) for word_id in sent]
        sent_vec = tf.reduce_mean(tf.stack(word_list, axis=0), axis=0)
        X.append(sent_vec)

    X = tf.stack(X, axis=0)
    P = tf.get_variable('P', (args.vec_dim, 4), dtype=tf.float32,
                        initializer=tf.random_uniform_initializer(-1., 1.))

    hypothesis = tf.matmul(X, P)
    classification_loss = tf.losses.softmax_cross_entropy(
        tf.one_hot(y, 4), hypothesis)
    return classification_loss


def get_full_loss(source_emb, target_emb, dictionary, X_id, y):
    """
    Compute the loss.
    $$ full_loss = alpha * classification_loss + (1 - alpha) * projection_loss $$

    dictionary: list[list[int]]
        the bilingual dictionary used to compute the projection loss, each item containing
        two indices.
    X_id: list[list[int]]
        examples in the sentiment dataset
    y: numpy.array of shape (datasetsize,)
        labels of the dataset
    """
    proj_loss = get_projection_loss(source_emb, target_emb, dictionary)
    classification_loss = get_classification_loss(source_emb, X_id, y)
    full_loss = tf.add(tf.multiply(args.alpha, classification_loss),
                       tf.multiply((1 - args.alpha), proj_loss))
    return full_loss


def get_projected_embeddings(source_original_emb, target_original_emb):
    """
    Given the original embeddings, calculate the projected word embeddings.
    """
    W_source = tf.get_variable(
        'W_source', (args.vec_dim, args.vec_dim), dtype=tf.float32, initializer=tf.random_uniform_initializer(-1., 1.))
    W_target = tf.get_variable(
        'W_target', (args.vec_dim, args.vec_dim), dtype=tf.float32, initializer=tf.random_uniform_initializer(-1., 1.))
    source_emb = tf.matmul(source_original_emb, W_source, name='map_source')
    target_emb = tf.matmul(target_original_emb, W_target, name='map_target')
    return source_emb, target_emb


def main():
    source_wordvec = utils.WordVecs(args.source_embedding)
    target_wordvec = utils.WordVecs(args.target_embedding)
    args.vec_dim = source_wordvec.vec_dim
    dictionary = utils.BilingualDict(args.dictionary).filter(
        lambda x: x[0] != '-').get_indexed_dictionary(source_wordvec, target_wordvec)

    senti_dataset = utils.SentimentDataset(
        args.sentiment_dataset).to_index(source_wordvec, target_wordvec)

    source_original_emb = tf.placeholder(
        tf.float32, shape=(None, args.vec_dim))
    target_original_emb = tf.placeholder(
        tf.float32, shape=(None, args.vec_dim))

    source_emb, target_emb = get_projected_embeddings(
        source_original_emb, target_original_emb)

    global_step = tf.Variable(0,
                              dtype=tf.int32,
                              trainable=False,
                              name='global_step')

    loss = get_full_loss(source_emb, target_emb,
                         dictionary, *senti_dataset.train)
    optimizer = tf.train.AdamOptimizer(args.alpha).minimize(loss, global_step=global_step)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(args.epochs):
            print('[echo %d] training......' % epoch)
            loss_, _ = sess.run([loss, optimizer], feed_dict={
                             source_original_emb: source_wordvec.embedding, target_original_emb: target_wordvec.embedding})
            print('[epoch %d]     loss: %d' % (epoch, loss_))
        saver.save(sess, './checkpoints/blse')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sl', '--source_lang',
                        help='source language: en/es/ca/eu (default: en)',
                        default='eu')
    parser.add_argument('-tl', '--target_lang',
                        help='target language: en/es/ca/eu (default: es)',
                        default='es')
    parser.add_argument('-a', '--alpha',
                        help="trade-off between projection and classification objectives (default: .001)",
                        default=0.001,
                        type=float)
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
    args = parser.parse_args()
    main()
