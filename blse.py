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


def softmax_layer(input):
    """
    compute $$ logits = input * P $$,
    where the shape of logits is (None, n_classes)
    doesn't perform softmax
    """
    with tf.variable_scope('softmax', reuse=tf.AUTO_REUSE):
        P = tf.get_variable('P', (args.vec_dim, 4), dtype=tf.float32,
                            initializer=tf.random_uniform_initializer(-1., 1.))
    return tf.matmul(input, P)


def get_classification_loss_pred(emb, X, y):
    """
    Given the language embedding and the classification dataset,
    compute the classification loss.
    """
    X = tf.nn.embedding_lookup(emb, X)  # shape: (None, 256, 300)
    X = tf.reduce_mean(X, axis=1)  # shape: (None, 300)
    # for sent in X_id:
    #     word_list = [tf.nn.embedding_lookup(emb, word_id) for word_id in sent]
    #     sent_vec = tf.reduce_mean(tf.stack(word_list, axis=0), axis=0)
    #     X.append(sent_vec)
    # X = tf.stack(X, axis=0)

    hypothesis = softmax_layer(X)
    pred = tf.argmax(hypothesis, axis=1, output_type=tf.int32)
    classification_loss = tf.losses.softmax_cross_entropy(
        tf.one_hot(y, 4), hypothesis)
    return classification_loss, pred


def get_full_loss_pred(source_emb, target_emb, dictionary, X_id, y):
    """
    Compute the loss and classification prediction.
    $$ full_loss = alpha * classification_loss + (1 - alpha) * projection_loss $$

    source_emb: tensor of shape (vocabsize, vec_dim)
        projected word embedding of the source language
    target_emb: tensor of shape (vocabsize, vec_dim)
        projected word embedding of the target language    
    dictionary: tensor of shape (dictsize, 2)
        the bilingual dictionary used to compute the projection loss, each item containing
        two indices.
    X_id: list[list[int]]
        examples in the sentiment dataset
    y: tensor of shape (None,)
        labels of the dataset
    """
    proj_loss = get_projection_loss(source_emb, target_emb, dictionary)
    classification_loss, pred = get_classification_loss_pred(
        source_emb, X_id, y)
    full_loss = args.alpha * classification_loss + (1 - args.alpha) * proj_loss
    return full_loss, classification_loss, proj_loss, pred


# def get_prediction(emb, X_id):
#     X = []
#     for sent in X_id:
#         word_list = [tf.nn.embedding_lookup(emb, word_id) for word_id in sent]
#         sent_vec = tf.reduce_mean(tf.stack(word_list, axis=0), axis=0)
#         X.append(sent_vec)
#     X = tf.stack(X, axis=0)
#     pred = tf.argmax(softmax_layer(X), axis=1)
#     return pred


def get_projected_embeddings(source_original_emb, target_original_emb):
    """
    Given the original embeddings, calculate the projected word embeddings.
    """
    with tf.variable_scope('projection', reuse=tf.AUTO_REUSE):
        W_source = tf.get_variable(
            'W_source', (args.vec_dim, args.vec_dim), dtype=tf.float32, initializer=tf.random_uniform_initializer(-1., 1.))
        W_target = tf.get_variable(
            'W_target', (args.vec_dim, args.vec_dim), dtype=tf.float32, initializer=tf.random_uniform_initializer(-1., 1.))
    source_emb = tf.matmul(source_original_emb, W_source, name='map_source')
    target_emb = tf.matmul(target_original_emb, W_target, name='map_target')
    return source_emb, target_emb


def load_data():
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


def get_input_layer():
    source_original_emb = tf.placeholder(
        tf.float32, shape=(None, args.vec_dim))
    target_original_emb = tf.placeholder(
        tf.float32, shape=(None, args.vec_dim))
    corpus = tf.placeholder(tf.int32, shape=(None, 256))
    labels = tf.placeholder(tf.int32, shape=(None,))
    dictionary = tf.placeholder(tf.int32, shape=(None, 2))
    return source_original_emb, target_original_emb, corpus, labels, dictionary


def main():
    source_emb_obj, target_emb_obj, dict_obj, train_x, train_y, test_x, test_y = load_data()  # numpy arrays

    source_original_emb, target_original_emb, corpus, labels, dictionary = get_input_layer()  # tensors

    source_emb, target_emb = get_projected_embeddings(
        source_original_emb, target_original_emb)

    global_step = tf.Variable(0,
                              dtype=tf.int32,
                              trainable=False,
                              name='global_step')
    loss, class_loss, proj_loss, pred = get_full_loss_pred(source_emb, target_emb,
                                                           dictionary, corpus, labels)

    acc = tf.reduce_mean(tf.to_float(tf.equal(pred, labels)))
    optimizer = tf.train.AdamOptimizer(
        args.learning_rate).minimize(loss, global_step=global_step)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(args.epochs):
            loss_, class_loss_, proj_loss_, _, train_acc_ = sess.run([loss, class_loss, proj_loss, optimizer, acc], feed_dict={
                source_original_emb: source_emb_obj,
                target_original_emb: target_emb_obj,
                corpus: train_x,
                labels: train_y,
                dictionary: dict_obj, })

            acc_ = sess.run(acc, feed_dict={
                source_original_emb: source_emb_obj,
                corpus: test_x,
                labels: test_y})
            print('epoch: %d  loss: %.4f  class_loss: %.4f  proj_loss_: %.4f  train_acc: %.2f  test_acc: %.2f' %
                  (epoch, loss_, class_loss_, proj_loss_, train_acc_, acc_))

            if (epoch + 1) % 10 == 0:
                saver.save(sess, './checkpoints/blse', global_step=global_step)


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
    args = parser.parse_args()
    main()
